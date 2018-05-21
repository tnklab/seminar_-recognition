# seminar_-recognition
認識についてのお話とサンプル
## 手順について[ライブラリのインストール編]
 - スライド11ページ
`$ sudo apt-get install tcsh`  
`$ sudo apt-get install python python-pip`  
`$ sudo pip install –upgrade pip`  
`$ sudo pip install numpy`  
`$ sudo pip install chainer==3.2.0`  
`$ wegt http://downloads.sourceforge.net/sp-tk/SPTK-3.5.tar.gz`  
`$ tar –zvxf SPTK-3.5.tar.gz`  
`$ cd SPTK-3.5`  
`$ ./configure`  
`$ make`  
`$ sudo make install`  

 - スライド12ページ  
`$ sudo apt-get install portaudio19-dev`  
`$ sudo apt-get install libblas-dev`  
`$ sudo apt-get install liblapack-dev`  
`$ sudo apt-get install gfortran`  
`$ sudo pip install scipy`  
`$ sudo apt-get install libportaudio-dev`  
`$ sudo apt-get install libatlas-base-dev`  
`$ sudo apt-get install python-pyaudio`  

## ファイルの書き加え
 - スライド14  
`$ sudo nano /etc/modprobe.d/alsa-base.conf`  
`options snd slots=snd_usb_audio,snd_bcm2835
options snd_usb_audio index=0
options snd_bcm2835 index=1`  
`$ sudo reboot now`  
`$ cat /proc/asound/modules`  


