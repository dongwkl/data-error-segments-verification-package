# Third-party notices

## Tennessee Eastman Process (TEP) code in `tep2py/indep_builds/build_1` to `build_12`

Each build folder holds the TEP Fortran source (`temain_mod.f`, `teprob.f`), the f2py interface
(`temain_mod-smart.pyf`), the compiled Python extension module (`temain_mod.cpython-311-x86_64-linux-gnu.so`),
the compilation log, and the normal-operation output `baseline.pkl` of that build. The builds differ only in the
random-number constant `G` in `teprob.f`.

The original code was written by James J. Downs and Ernest F. Vogel (Tennessee Eastman Company). The modified
version (`temain_mod.f`) is by Evan L. Russell, Leo H. Chiang, and Richard D. Braatz (Large Scale Systems Research
Laboratory, University of Illinois at Urbana-Champaign) and carries the following notice, reproduced here as
the notice requires for redistribution in source and binary form:

```text
The modified text is Copyright 1998-2002 by The Board of Trustees
of the University of Illinois.  All rights reserved.

Permission hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:
		1. Redistributions of source code must retain the above copyright
	   notice, this list of conditions and the following disclaimers.
	2. Redistributions in binary form must reproduce the above
	   copyright notice, this list of conditions and the following
	   disclaimers in the documentation and/or other materials
	   provided with the distribution.
	3. Neither the names of Large Scale Research Systems Laboratory,
	   University of Illinois, nor the names of its contributors may
	   be used to endorse or promote products derived from this
	   Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

`teprob.f` carries the original authorship notice of Downs and Vogel and no separate license text. We have
not checked the terms of use of that file beyond this notice.

The simulator is used through the interface of tep2py (Mauricio camaramm, GitHub user camaramm; listed there under the MIT license).
The Fortran code was modified by Xavier and de Seixas (2018), whose paper `temain_mod.f` asks users to cite. The compiled
modules in this package were built from the Fortran sources with f2py and gfortran 13 (see `build.log` in each build folder).

### References requested by the code authors

```text
Users should cite the original code using the following references:

   J.J. Downs and E.F. Vogel, "A plant-wide industrial process control
   problem." Presented at the AIChE 1990 Annual Meeting, Session on
   Industrial Challenge Problems in Process Control, Paper #24a
   Chicago, Illinois, November 14, 1990.

   J.J. Downs and E.F. Vogel, "A plant-wide industrial process control
   problem," Computers and Chemical Engineering, 17:245-255 (1993).

 Users should cite the modified code using the following references:

   E.L. Russell, L.H. Chiang, and R.D. Braatz. Data-driven Techniques
   for Fault Detection and Diagnosis in Chemical Processes, Springer-Verlag,
   London, 2000.

   L.H. Chiang, E.L. Russell, and R.D. Braatz. Fault Detection and
   Diagnosis in Industrial Systems, Springer-Verlag, London, 2001.

   L.H. Chiang, E.L. Russell, and R.D. Braatz. "Fault diagnosis in
   chemical processes using Fisher discriminant analysis, discriminant
   partial least squares, and principal component analysis," Chemometrics
   and Intelligent Laboratory Systems, 50:243-252, 2000.
```

The following works are also cited in the accompanying manuscript:

- Downs, J. J., and Vogel, E. F. (1993), A plant-wide industrial process control problem, Computers and Chemical Engineering, 17(3), 245-255.
- Bathelt, A., Ricker, N. L., and Jelali, M. (2015), Revision of the Tennessee Eastman process model, IFAC-PapersOnLine, 48(8), 309-314.
- Xavier, G. M., and de Seixas, J. M. (2018), Fault detection and diagnosis in a chemical process using long short-term memory recurrent neural network, 2018 International Joint Conference on Neural Networks (IJCNN), IEEE, 1-8.
- Mauricio camaramm (GitHub user camaramm), tep2py: A Python interface to the Tennessee Eastman Process Fortran program, GitHub repository, https://github.com/camaramm/tep2py (accessed 4 November 2026).
