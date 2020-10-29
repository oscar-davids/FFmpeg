FFmpeg README
=============

FFmpeg is a collection of libraries and tools to process multimedia content
such as audio, video, subtitles and related metadata.

## Libraries

* `libavcodec` provides implementation of a wider range of codecs.
* `libavformat` implements streaming protocols, container formats and basic I/O access.
* `libavutil` includes hashers, decompressors and miscellaneous utility functions.
* `libavfilter` provides a mean to alter decoded Audio and Video through chain of filters.
* `libavdevice` provides an abstraction to access capture and playback devices.
* `libswresample` implements audio mixing and resampling routines.
* `libswscale` implements color conversion and scaling routines.

## Tools

* [ffmpeg](https://ffmpeg.org/ffmpeg.html) is a command line toolbox to
  manipulate, convert and stream multimedia content.
* [ffplay](https://ffmpeg.org/ffplay.html) is a minimalistic multimedia player.
* [ffprobe](https://ffmpeg.org/ffprobe.html) is a simple analysis tool to inspect
  multimedia content.
* Additional small tools such as `aviocat`, `ismindex` and `qt-faststart`.

## Documentation

The offline documentation is available in the **doc/** directory.

The online documentation is available in the main [website](https://ffmpeg.org)
and in the [wiki](https://trac.ffmpeg.org).

### Examples

Coding examples are available in the **doc/examples** directory.

## License

FFmpeg codebase is mainly LGPL-licensed with optional components licensed under
GPL. Please refer to the LICENSE file for detailed information.

## Contributing

Patches should be submitted to the ffmpeg-devel mailing list using
`git format-patch` or `git send-email`. Github pull requests should be
avoided because they are not part of our review process and will be ignored.




FFmpeg with lvpdiff filter
=============

## Build FFmpeg with lvpdiff filter on ubuntu18.04

FFmpeg is a collection of libraries and tools to process multimedia content
such as audio, video, subtitles and related metadata.

1. prerequest

These are packages required for compiling, but you can remove them when you are done if you prefer:

- install pre-built packages

	sudo apt-get update -qq && sudo apt-get -y install autoconf automake build-essential cmake \
  git-core libass-dev libfreetype6-dev libsdl2-dev libtool libva-dev libvdpau-dev libvorbis-dev\
  libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo wget zlib1g-dev

	sudo apt-get install nasm

	sudo apt-get install yasm
	
	#to build opencv
	
	sudo apt-get install cmake-gui

To build ffmpeg with lvpdiff filter, need to install customized opencv. 
Now opencv support to build on ubuntu GUI, so need ubuntu os with GUI.

- build and install OpenCV Library

  clone source code 
  
	git clone https://github.com/oscar-davids/opencv
  
  git checkout dev-lvpdiff

  build and install openCV
  
  cd opencv
  
  mkdir build
  
  cd build
  
  cmake-gui ..
    
  ![Alt text](ubuntu-cmake-gui-setting.png?raw=true "")

  #in build directory
  
  sudo make -j8
  
  #install to /usr/local/bin
  
  sudo make install
  
2. FFmpeg Compilation 

- clone source code 

  git clone https://github.com/oscar-davids/FFmpeg
  
  git checkout dev-lvpdiff
  
- build 

  ./configure --prefix="$HOME/ffmpeg_build" --enable-shared --enable-libopencv

  make && make install
  
3. execute
 
  cd $HOME/ffmpeg_build/bin
  
  ffmpeg -i main.mp4 -i ref.mp4 -lavfi  "lvpdiff" -f null -
  
  ffmpeg -i main.mp4 -i ref.mp4 -lavfi  lvpdiff="stats_file=stats.log" -f null -
  
  **remark some case need to set env:  export LD_LIBRARY_PATH="$HOME/ffmpeg_build/lib/:/usr/local/lib:$LD_LIBRARY_PATH"
 
 ## Build FFmpeg with lvpdiff_cuda filter on ubuntu18.04

1. prerequest

  - install nvidia cuda driver and SDK
  
  	https://developer.nvidia.com/cuda-downloads
  
  - install pre-built packages (same as above)  
  
  - build and install OpenCV CUDA Library
  
	  clone source code 

		git clone https://github.com/oscar-davids/opencv

	  git checkout dev-lvpdiff-cuda

	  build and install openCV

	  cd opencv

	  mkdir build

	  cd build

	  cmake-gui ..

	  ![Alt text](ubuntu-cmake-gui-setting.png?raw=true "")

	  #in build directory

	  sudo make -j8

	  #install to /usr/local/bin

	  sudo make install
	  
2. FFmpeg Compilation 

- clone source code 

  git clone https://github.com/oscar-davids/FFmpeg
  
  git checkout dev-lvpdiff-cuda
  
- build 

  ./configure --prefix="$HOME/ffmpeg_build" --enable-shared --enable-libopencv --enable-libopencv-cuda

  make && make install
  
3. execute
 
  cd $HOME/ffmpeg_build/bin
  
  ffmpeg -i main.mp4 -i ref.mp4 -lavfi  "lvpdiff" -f null -
  
  ffmpeg -i main.mp4 -i ref.mp4 -lavfi  lvpdiff="calcmode=cuda:stats_file=stats.log" -f null -
  
  ** calcmode(skip, cpu, cuda)  
  
  **remark some case need to set env:  export LD_LIBRARY_PATH="$HOME/ffmpeg_build/lib/:/usr/local/lib:$LD_LIBRARY_PATH"  
  
  
## Build FFmpeg with lvpdiff filter on Windows10

1. prerequest

  Visual Studio 2017 or 2019
  MSYS2
  YASM

   Visual Studio
   
   https://visualstudio.microsoft.com/
   
   MSYS2
   
    Download and run the installer at linked URL(http://msys2.github.io, https://packages.msys2.org/package/mingw-w64-x86_64-gcc). 
    
    Follow the instructions and install it in "C:\workspace\windows\msys64" folder.
    
    open MSYS2 MSYS(x64) windows console as administrator
    
    Install required tools: 
    
	    pacman -S make gcc diffutils git
      
    Rename C:\workspace\windows\msys64\usr\bin\link.exe to C:\workspace\windows\msys64\usr\bin\link_orig.exe, in order to use MSVC  link.exe (naming conflict)
    
    remove "rem" comment in "c:/msys64/msys2_shell.cmd" file.

	rem set MSYS2_PATH_TYPE=inherit -> set MSYS2_PATH_TYPE=inherit

   YASM
   
    Download Win64.exe(http://yasm.tortall.net/Download.html) and move it to C:\workspace\windows
    
    Rename yasm-<version>-win64.exe to yasm.exe
    
    Add C:\workspace\windows to PATH environment variable (setx PATH "%PATH%;C:\workspace\windows)
    
    Ready to compile
    
- Install nvidia driver and cuda SDK(option)

    https://developer.nvidia.com/cuda-toolkit

    https://developer.nvidia.com/rdp/cudnn-download    
  
- Install OpenCV 

  clone source code 
  
	git clone https://github.com/oscar-davids/opencv
  
  git checkout dev-lvpdiff or (dev-lvpdiff-cuda)

  build and install openCV using cmake-gui and visualstudio compile tool   

2. Compilation 

- Launch msys2 shell from Visual Studio Code shell
  
  open console
  
  	C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvars64.bat
  
  set tensorflow library path
  
  	set lib=%lib%;c:/$opencvlibpath
  
  Execute msys2
  
  	C:\workspace\windows\msys64\msys2_shell.cmd -mingw64
  
  Following command have to execute in new msys2 console winndow.
  
- copy openCV c headfiles and lib

 	cp  -r $opencvroot/include/ /usr/local/include
	
	cp  -r $opencvroot/lib/opencv_core.lib opencv_imageproc.lib /usr/local/lib/
	
	(cuda option) cp  -r $opencvroot/lib/*.lib  /usr/local/lib/
		  
- clone source code 

  git clone https://github.com/oscar-davids/FFmpeg
  
  git checkout dev-lvpdiff or dev-lvpdiff-cuda
  
- copy cuda c headfiles and lib(option)

  	cd FFmpeg    
	
	./configure --toolchain=msvc --arch=x86_64 --enable-libopencv --enable-shared --extra-cflags=-I/usr/local/include --extra-ldflags='-LIBPATH:/usr/lib' --extra-ldflags='-LIBPATH:/usr/local/lib'
	
	(cuda option) --enable-libopencv-cuda

	make -j 8
	
3. execute

  	ffmpeg -i main.mp4 -i ref.mp4 -lavfi  "lvpdiff" -f null -
  
  	ffmpeg -i main.mp4 -i ref.mp4 -lavfi  lvpdiff="stats_file=stats.log" -f null -
	
	ffmpeg -i main.mp4 -i ref.mp4 -lavfi  lvpdiff="calcmode=cuda:stats_file=stats.log" -f null -
  

Reference
=============

https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu


