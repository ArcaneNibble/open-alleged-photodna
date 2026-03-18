# PhotoDNA leaked binary blob harness

There is now a somewhat-hacky harness which invokes the leaked PhotoDNA binary blob (specifically on not-Windows) and automatically dumps intermediate values for comparison. (You of course still need the ability to somehow execute x86-64 code.)

This allows testing and further validating that this code produces the correct results.

There is relatively little code in the PhotoDNA DLL, and it doesn't actually require any operating system functionality other than `malloc` and `free`. Thus it is quite possible to "just" load the code on platforms other than Windows, and then call it. This allows testing on macOS and Linux.

This harness does the _absolute_ minimum. It loads the code (`.text`) and read-only data (`.rdata`) sections. It doesn't resolve any DLL imports. It emulates `malloc` and `free`, and it patches out just enough stack-probe and stack-cookie checking so that the main function runs.

It also generates a number of thunks which convert between the Windows x86-64 [ABI](https://en.wikipedia.org/wiki/X86_calling_conventions#x86-64_calling_conventions) and the macOS/Linux one.

In order to run this on ARM64 macOS, you will need an x86-64 version of Python running under Rosetta 2. The recommended way to get this is to use the _official_ Python `.pkg` installers which are "fat binaries" containing both architectures. `arch -x86_64 ./binary-harness.py` forces x86-64 mode.
