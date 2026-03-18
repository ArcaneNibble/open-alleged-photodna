#!/usr/bin/env python3

# This code does two major things:
# 1. Loads the PhotoDNA binary DLL on platforms that are *not* Windows
#    (assumes SysV x86_64 ABI)
# 2. Applies hacks to the DLL so that we can dump intermediate state

import ctypes
import errno
import hashlib
import mmap
import os
import pefile
import platform
import sys
from PIL import Image

import oaphotodna

assert platform.machine() == 'x86_64', "Must be running on x86_64 (use `arch -x86_64` on macOS)"
assert os.name == 'posix', "Windows is *NOT* supported"

VALIDATE_BINARY = True
REFERENCE_BINARY_HASH = 'b91f77124065ae7d7c3cbd382d7cf8ab8283af4a942aff3fd9fdacd55af08091'
REFERENCE_BINARY_FILENAME = "PhotoDNAx64.dll"
if VALIDATE_BINARY:
    with open(REFERENCE_BINARY_FILENAME, 'rb') as f:
        ref_binary = f.read()
    assert hashlib.sha256(ref_binary).hexdigest() == REFERENCE_BINARY_HASH

DEBUG_LOGGING = False


def divroundup(val, div):
    return (val + div - 1) // div * div

# ----- Callback functions -----

@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint64)
def _malloc(sz):
    malloc_ty = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint64)
    malloc = malloc_ty(('malloc', ctypes.CDLL(None)))
    ptr = malloc(sz)
    if DEBUG_LOGGING:
        print(f"MALLOC! {sz:016x} -> {ptr:016x}")
    return ptr


@ctypes.CFUNCTYPE(None, ctypes.c_uint64)
def _free(ptr):
    free_ty = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    free = free_ty(('free', ctypes.CDLL(None)))
    free(ptr)
    if DEBUG_LOGGING:
        print(f"FREE! {ptr:016x}")

# The following hooks take `rbp` as their parameter

@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def hook_after_feature(ptr):
    if DEBUG_LOGGING:
        print(f"hook after feature! {ptr:016x}")
    vals = (ctypes.c_double * (26 * 26)).from_address(ptr + 0x7f0)
    vals = list(vals)
    if DEBUG_LOGGING:
        print(vals)
    global _vals_after_feature
    _vals_after_feature = vals


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def hook_after_grad(ptr):
    if DEBUG_LOGGING:
        print(f"hook after gradient! {ptr:016x}")
    vals = (ctypes.c_double * (6 * 6 * 4)).from_address(ptr + 0x370)
    vals = list(vals)
    if DEBUG_LOGGING:
        print(vals)
    global _vals_after_grad
    _vals_after_grad = vals


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def hook_after_hash(ptr):
    if DEBUG_LOGGING:
        print(f"hook after hash! {ptr:016x}")
    vals = (ctypes.c_double * (6 * 6 * 4)).from_address(ptr + 0x370)
    vals = list(vals)
    if DEBUG_LOGGING:
        print(vals)
    global _vals_after_hash
    _vals_after_hash = vals


def load_dll():
    # Do basic parsing junk
    pe = pefile.PE(REFERENCE_BINARY_FILENAME)

    text_section = pe.sections[0]
    rdata_section = pe.sections[1]
    if DEBUG_LOGGING:
        print(pe.OPTIONAL_HEADER)
        print(text_section)
        print(rdata_section)

    pe_file = open(REFERENCE_BINARY_FILENAME, 'rb')

    virt_align = pe.OPTIONAL_HEADER.SectionAlignment
    text_to_rdata_sz = rdata_section.VirtualAddress - text_section.VirtualAddress
    tot_sz = text_to_rdata_sz + divroundup(rdata_section.Misc_VirtualSize, virt_align)
    # This extra dummy page allows __security_cookie to not crash
    # (it's technically the beginning of `.data`, but *none* of that data is actually needed/used)
    tot_sz += virt_align

    exports = {}
    for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
        if DEBUG_LOGGING:
            print(hex(exp.address - text_section.VirtualAddress), exp.name, exp.ordinal)
        exports[exp.name] = exp.address - text_section.VirtualAddress

    # Reserve memory
    reserved_addr = mmap.mmap(
        -1, tot_sz,
        mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        mmap.PROT_READ | mmap.PROT_WRITE)
    reserved_addr_buf = (ctypes.c_uint8 * tot_sz).from_buffer(reserved_addr)
    reserved_addr_ptr = ctypes.addressof(reserved_addr_buf)
    if DEBUG_LOGGING:
        print(f"dll loaded @ 0x{reserved_addr_ptr:016x}")
    assert reserved_addr_ptr & (virt_align - 1) == 0, "bad alignment"

    # Map .text
    pe_file.seek(text_section.PointerToRawData)
    reserved_addr_buf[:text_section.Misc_VirtualSize] = pe_file.read(text_section.Misc_VirtualSize)

    # Map .rdata
    pe_file.seek(rdata_section.PointerToRawData)
    reserved_addr_buf[text_to_rdata_sz:text_to_rdata_sz+rdata_section.Misc_VirtualSize] = \
        pe_file.read(rdata_section.Misc_VirtualSize)

    pe_file.close()

    # Patches

    # malloc    @ 0x19000 + 0x9400
    i = 0x19000 + 0x9400

    def pushb(b):
        nonlocal i
        reserved_addr_buf[i] = b
        i += 1

    pushb(0x56)                                                         # push rsi
    pushb(0x57)                                                         # push rdi
    pushb(0x48), pushb(0x89), pushb(0xCF)                               # mov rdi, rcx
    pushb(0x48), pushb(0xb8)                                            # movabs rax, <xxx>
    malloc_addr = ctypes.cast(_malloc, ctypes.c_void_p).value
    pushb((malloc_addr >> 0) & 0xff)
    pushb((malloc_addr >> 8) & 0xff)
    pushb((malloc_addr >> 16) & 0xff)
    pushb((malloc_addr >> 24) & 0xff)
    pushb((malloc_addr >> 32) & 0xff)
    pushb((malloc_addr >> 40) & 0xff)
    pushb((malloc_addr >> 48) & 0xff)
    pushb((malloc_addr >> 56) & 0xff)
    pushb(0xFF), pushb(0xD0)                                            # call rax
    pushb(0x5F)                                                         # pop rdi
    pushb(0x5E)                                                         # pop rsi
    pushb(0xC3)                                                         # ret
    # patch thunk jmp to _malloc_base
    jump_offs = 0x19000 + 0x9400 - (0x10e08 + 5)
    reserved_addr_buf[0x10e09+0] = (jump_offs >> 0) & 0xff
    reserved_addr_buf[0x10e09+1] = (jump_offs >> 8) & 0xff
    reserved_addr_buf[0x10e09+2] = (jump_offs >> 16) & 0xff
    reserved_addr_buf[0x10e09+3] = (jump_offs >> 24) & 0xff

    # free    @ 0x19000 + 0x9500
    i = 0x19000 + 0x9500
    pushb(0x56)                                                         # push rsi
    pushb(0x57)                                                         # push rdi
    pushb(0x48), pushb(0x89), pushb(0xCF)                               # mov rdi, rcx
    pushb(0x48), pushb(0xb8)                                            # movabs rax, <xxx>
    free_addr = ctypes.cast(_free, ctypes.c_void_p).value
    pushb((free_addr >> 0) & 0xff)
    pushb((free_addr >> 8) & 0xff)
    pushb((free_addr >> 16) & 0xff)
    pushb((free_addr >> 24) & 0xff)
    pushb((free_addr >> 32) & 0xff)
    pushb((free_addr >> 40) & 0xff)
    pushb((free_addr >> 48) & 0xff)
    pushb((free_addr >> 56) & 0xff)
    pushb(0xFF), pushb(0xD0)                                            # call rax
    pushb(0x5F)                                                         # pop rdi
    pushb(0x5E)                                                         # pop rsi
    pushb(0xC3)                                                         # ret
    # patch thunk jmp to _free_base
    jump_offs = 0x19000 + 0x9500 - (0x10e00 + 5)
    reserved_addr_buf[0x10e01+0] = (jump_offs >> 0) & 0xff
    reserved_addr_buf[0x10e01+1] = (jump_offs >> 8) & 0xff
    reserved_addr_buf[0x10e01+2] = (jump_offs >> 16) & 0xff
    reserved_addr_buf[0x10e01+3] = (jump_offs >> 24) & 0xff

    # ABI thunk for ComputeRobustHash   @ 0x19000 + 0x9600
    i = 0x19000 + 0x9600
    pushb(0x55)                                                         # push rbp
    pushb(0x48), pushb(0x89), pushb(0xE5)                               # mov rbp, rsp
    pushb(0x41), pushb(0x51)                                            # push r9
    pushb(0x41), pushb(0x50)                                            # push r8
    pushb(0x48), pushb(0x83), pushb(0xEC), pushb(0x20)                  # sub rsp, 0x20
    pushb(0x51)                                                         # push rcx
    pushb(0x52)                                                         # push rdx
    pushb(0x48), pushb(0x89), pushb(0xF9)                               # mov rcx, rdi
    pushb(0x48), pushb(0x89), pushb(0xF2)                               # mov rdx, rsi
    pushb(0x41), pushb(0x58)                                            # pop r8
    pushb(0x41), pushb(0x59)                                            # pop r9
    call_offs = (exports[b'ComputeRobustHash'] - (i + 5)) & 0xffffffff
    pushb(0xE8),                                                        # call <xxx>
    pushb((call_offs >> 0) & 0xff)
    pushb((call_offs >> 8) & 0xff)
    pushb((call_offs >> 16) & 0xff)
    pushb((call_offs >> 24) & 0xff)
    pushb(0xC9)                                                         # leave
    pushb(0xC3)                                                         # ret

    # Hook after feature grid is computed   @ 0x19000 + 0x9700
    i = 0x19000 + 0x9700
    pushb(0x51)                                                         # push rcx
    pushb(0x52)                                                         # push rdx
    pushb(0x56)                                                         # push rsi
    pushb(0x57)                                                         # push rdi
    pushb(0x41), pushb(0x50)                                            # push r8
    pushb(0x41), pushb(0x51)                                            # push r9
    pushb(0x41), pushb(0x52)                                            # push r10
    pushb(0x41), pushb(0x53)                                            # push r11
    pushb(0x48), pushb(0x81), pushb(0xec)                               # sub rsp, 512
    pushb(0x00), pushb(0x02), pushb(0x00), pushb(0x00)
    pushb(0x0f), pushb(0xae), pushb(0x04), pushb(0x24)                  # fxsave [rsp]
    pushb(0x48), pushb(0x89), pushb(0xef)                               # mov rdi, rbp
    pushb(0x48), pushb(0xb8)                                            # movabs rax, <xxx>
    hook_after_feature_addr = ctypes.cast(hook_after_feature, ctypes.c_void_p).value
    pushb((hook_after_feature_addr >> 0) & 0xff)
    pushb((hook_after_feature_addr >> 8) & 0xff)
    pushb((hook_after_feature_addr >> 16) & 0xff)
    pushb((hook_after_feature_addr >> 24) & 0xff)
    pushb((hook_after_feature_addr >> 32) & 0xff)
    pushb((hook_after_feature_addr >> 40) & 0xff)
    pushb((hook_after_feature_addr >> 48) & 0xff)
    pushb((hook_after_feature_addr >> 56) & 0xff)
    pushb(0xff), pushb(0xd0)                                            # call rax
    pushb(0x0f), pushb(0xae), pushb(0x0c), pushb(0x24)                  # fxrstor [rsp]
    pushb(0x48), pushb(0x81), pushb(0xc4)                               # add rsp, 512
    pushb(0x00), pushb(0x02), pushb(0x00), pushb(0x00)
    pushb(0x41), pushb(0x5b)                                            # pop r11
    pushb(0x41), pushb(0x5a)                                            # pop r10
    pushb(0x41), pushb(0x59)                                            # pop r9
    pushb(0x41), pushb(0x58)                                            # pop r8
    pushb(0x5f)                                                         # pop rdi
    pushb(0x5e)                                                         # pop rsi
    pushb(0x5a)                                                         # pop rdx
    pushb(0x59)                                                         # pop rcx
    jump_offs = (0xa199 - (i + 5)) & 0xffffffff
    pushb(0xe9)                                                         # jmp <xxx>
    pushb((jump_offs >> 0) & 0xff)
    pushb((jump_offs >> 8) & 0xff)
    pushb((jump_offs >> 16) & 0xff)
    pushb((jump_offs >> 24) & 0xff)
    # Apply the hook (patches a je opcode to jump somewhere else)
    jump_offs = (0x19000 + 0x9700 - (0x8907 + 6)) & 0xffffffff
    reserved_addr_buf[0x8907 + 2] = (jump_offs >> 0) & 0xff
    reserved_addr_buf[0x8907 + 3] = (jump_offs >> 8) & 0xff
    reserved_addr_buf[0x8907 + 4] = (jump_offs >> 16) & 0xff
    reserved_addr_buf[0x8907 + 5] = (jump_offs >> 24) & 0xff

    # Hook after gradient grid is computed      @ 0x19000 + 0x9800
    i = 0x19000 + 0x9800
    pushb(0x51)                                                         # push rcx
    pushb(0x52)                                                         # push rdx
    pushb(0x56)                                                         # push rsi
    pushb(0x57)                                                         # push rdi
    pushb(0x41), pushb(0x50)                                            # push r8
    pushb(0x41), pushb(0x51)                                            # push r9
    pushb(0x41), pushb(0x52)                                            # push r10
    pushb(0x41), pushb(0x53)                                            # push r11
    pushb(0x48), pushb(0x81), pushb(0xec)                               # sub rsp, 512
    pushb(0x00), pushb(0x02), pushb(0x00), pushb(0x00)
    pushb(0x0f), pushb(0xae), pushb(0x04), pushb(0x24)                  # fxsave [rsp]
    pushb(0x48), pushb(0x89), pushb(0xef)                               # mov rdi, rbp
    pushb(0x48), pushb(0xb8)                                            # movabs rax, <xxx>
    hook_after_grad_addr = ctypes.cast(hook_after_grad, ctypes.c_void_p).value
    pushb((hook_after_grad_addr >> 0) & 0xff)
    pushb((hook_after_grad_addr >> 8) & 0xff)
    pushb((hook_after_grad_addr >> 16) & 0xff)
    pushb((hook_after_grad_addr >> 24) & 0xff)
    pushb((hook_after_grad_addr >> 32) & 0xff)
    pushb((hook_after_grad_addr >> 40) & 0xff)
    pushb((hook_after_grad_addr >> 48) & 0xff)
    pushb((hook_after_grad_addr >> 56) & 0xff)
    pushb(0xff), pushb(0xd0)                                            # call rax
    pushb(0x0f), pushb(0xae), pushb(0x0c), pushb(0x24)                  # fxrstor [rsp]
    pushb(0x48), pushb(0x81), pushb(0xc4)                               # add rsp, 512
    pushb(0x00), pushb(0x02), pushb(0x00), pushb(0x00)
    pushb(0x41), pushb(0x5b)                                            # pop r11
    pushb(0x41), pushb(0x5a)                                            # pop r10
    pushb(0x41), pushb(0x59)                                            # pop r9
    pushb(0x41), pushb(0x58)                                            # pop r8
    pushb(0x5f)                                                         # pop rdi
    pushb(0x5e)                                                         # pop rsi
    pushb(0x5a)                                                         # pop rdx
    pushb(0x59)                                                         # pop rcx
    jump_offs = (0xb04b - (i + 5)) & 0xffffffff
    pushb(0xe9)                                                         # jmp <xxx>
    pushb((jump_offs >> 0) & 0xff)
    pushb((jump_offs >> 8) & 0xff)
    pushb((jump_offs >> 16) & 0xff)
    pushb((jump_offs >> 24) & 0xff)
    # Apply the hook (patches a jne to an unconditional jump (breaking hyperparameter values other than 6))
    jump_offs = (0x19000 + 0x9800 - (0xb045 + 6)) & 0xffffffff
    reserved_addr_buf[0xb045 + 0] = 0x90
    reserved_addr_buf[0xb045 + 1] = 0xe9
    reserved_addr_buf[0xb045 + 2] = (jump_offs >> 0) & 0xff
    reserved_addr_buf[0xb045 + 3] = (jump_offs >> 8) & 0xff
    reserved_addr_buf[0xb045 + 4] = (jump_offs >> 16) & 0xff
    reserved_addr_buf[0xb045 + 5] = (jump_offs >> 24) & 0xff

    # Hook after hash is computed as floats     @ 0x19000 + 0x9900
    i = 0x19000 + 0x9900
    pushb(0x51)                                                         # push rcx
    pushb(0x52)                                                         # push rdx
    pushb(0x56)                                                         # push rsi
    pushb(0x57)                                                         # push rdi
    pushb(0x41), pushb(0x50)                                            # push r8
    pushb(0x41), pushb(0x51)                                            # push r9
    pushb(0x41), pushb(0x52)                                            # push r10
    pushb(0x41), pushb(0x53)                                            # push r11
    pushb(0x48), pushb(0x81), pushb(0xec)                               # sub rsp, 512
    pushb(0x00), pushb(0x02), pushb(0x00), pushb(0x00)
    pushb(0x0f), pushb(0xae), pushb(0x04), pushb(0x24)                  # fxsave [rsp]
    pushb(0x48), pushb(0x89), pushb(0xef)                               # mov rdi, rbp
    pushb(0x48), pushb(0xb8)                                            # movabs rax, <xxx>
    hook_after_hash_addr = ctypes.cast(hook_after_hash, ctypes.c_void_p).value
    pushb((hook_after_hash_addr >> 0) & 0xff)
    pushb((hook_after_hash_addr >> 8) & 0xff)
    pushb((hook_after_hash_addr >> 16) & 0xff)
    pushb((hook_after_hash_addr >> 24) & 0xff)
    pushb((hook_after_hash_addr >> 32) & 0xff)
    pushb((hook_after_hash_addr >> 40) & 0xff)
    pushb((hook_after_hash_addr >> 48) & 0xff)
    pushb((hook_after_hash_addr >> 56) & 0xff)
    pushb(0xff), pushb(0xd0)                                            # call rax
    pushb(0x0f), pushb(0xae), pushb(0x0c), pushb(0x24)                  # fxrstor [rsp]
    pushb(0x48), pushb(0x81), pushb(0xc4)                               # add rsp, 512
    pushb(0x00), pushb(0x02), pushb(0x00), pushb(0x00)
    pushb(0x41), pushb(0x5b)                                            # pop r11
    pushb(0x41), pushb(0x5a)                                            # pop r10
    pushb(0x41), pushb(0x59)                                            # pop r9
    pushb(0x41), pushb(0x58)                                            # pop r8
    pushb(0x5f)                                                         # pop rdi
    pushb(0x5e)                                                         # pop rsi
    pushb(0x5a)                                                         # pop rdx
    pushb(0x59)                                                         # pop rcx
    pushb(0x4c), pushb(0x8b), pushb(0x85)                               # mov r8, qword [rbp+0x108]
    pushb(0x08), pushb(0x01), pushb(0x00), pushb(0x00)
    jump_offs = (0xd8c5 - (i + 5)) & 0xffffffff
    pushb(0xe9)                                                         # jmp <xxx>
    pushb((jump_offs >> 0) & 0xff)
    pushb((jump_offs >> 8) & 0xff)
    pushb((jump_offs >> 16) & 0xff)
    pushb((jump_offs >> 24) & 0xff)
    # Apply the hook (replaces a mov opcode, which is replicated in the code blob above)
    jump_offs = (0x19000 + 0x9900 - (0xd8be + 5)) & 0xffffffff
    reserved_addr_buf[0xd8be + 0] = 0xe9
    reserved_addr_buf[0xd8be + 1] = (jump_offs >> 0) & 0xff
    reserved_addr_buf[0xd8be + 2] = (jump_offs >> 8) & 0xff
    reserved_addr_buf[0xd8be + 3] = (jump_offs >> 16) & 0xff
    reserved_addr_buf[0xd8be + 4] = (jump_offs >> 24) & 0xff
    reserved_addr_buf[0xd8be + 5] = 0xcc
    reserved_addr_buf[0xd8be + 6] = 0xcc

    # Patch out __chkstk
    reserved_addr_buf[0xf010] = 0xc3
    # Patch out __security_check_cookie
    reserved_addr_buf[0xf080] = 0xc3

    # Set perms correctly
    mprotect_ty = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, use_errno=True)
    mprotect = mprotect_ty(('mprotect', ctypes.CDLL(None)))

    # NOTE: We set the whole thing as R+X, since our patches live *after* the end of .rdata
    _ret = mprotect(
        reserved_addr_ptr,
        tot_sz,
        mmap.PROT_READ | mmap.PROT_EXEC)
    assert _ret == 0, errno.errorcode[ctypes.get_errno()]

    return (reserved_addr, reserved_addr_ptr)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} image.jpg")
        sys.exit(-1)

    # Load the image
    im = Image.open(sys.argv[1], 'r')
    if im.mode != 'RGB':
        im = im.convert(mode='RGB')

    # "Load" the DLL
    (_photodna, photodna_loaded_addr) = load_dll()

    # Prepare to invoke the DLL
    ComputeRobustHash_ty = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_void_p)
    ComputeRobustHash = ComputeRobustHash_ty(photodna_loaded_addr + 0x19000 + 0x9600)

    # Invoke the DLL to get the reference data
    hashByteArray = (ctypes.c_ubyte * 144)()
    ComputeRobustHash(ctypes.c_char_p(im.tobytes()), im.width, im.height, 0, hashByteArray, 0)

    hashPtr = ctypes.cast(hashByteArray, ctypes.POINTER(ctypes.c_ubyte))
    reference_hash = [hashPtr[i] for i in range(144)]

    # Check ours against the reference
    summed_pixels = oaphotodna.preprocess_pixel_sum_(im)
    (feature_grid, grid_step_h, grid_step_v) = \
        oaphotodna.compute_feature_grid(summed_pixels, im.width, im.height)
    for i in range(len(feature_grid)):
        if feature_grid[i] != _vals_after_feature[i]:
            print(f"Feature grid compare failed @[{i}] expected {_vals_after_feature[i]} ours {feature_grid[i]}")
    gradient_grid = oaphotodna.compute_gradient_grid(feature_grid)
    for i in range(len(gradient_grid)):
        if gradient_grid[i] != _vals_after_grad[i]:
            print(f"Gradient grid compare failed @[{i}] expected {_vals_after_grad[i]} ours {gradient_grid[i]}")
    hash_as_floats = oaphotodna.process_hash(gradient_grid, grid_step_h, grid_step_v)
    for i in range(len(hash_as_floats)):
        if hash_as_floats[i] != _vals_after_hash[i]:
            print(f"Hash compare failed @[{i}] expected {_vals_after_hash[i]} ours {hash_as_floats[i]}")
    hash_as_bytes = oaphotodna.hash_to_bytes(hash_as_floats)
    for i in range(len(hash_as_bytes)):
        if hash_as_bytes[i] != reference_hash[i]:
            print(f"Hash (bytes) compare failed @[{i}] expected {_vals_after_hash[i]} ours {hash_as_bytes[i]}")


if __name__ == '__main__':
    main()
