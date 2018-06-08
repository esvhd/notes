# System

* Intel Xeon E5-2630v4
* Asus X99-E WS motherboard
* 2x Nvidia GeForce 1080 Ti Founders

# Ubuntu Setup Notes

I largely followed this [post](https://blog.slavv.com/the-1700-great-deep-learning-box-assembly-setup-and-benchmarks-148c5ebe6415) for the installation and software configuration instructions.

A few additions are needed, perhaps specifically for the hardware I'm using.

# Partitions

* `/` root drive on SSD drive
* `/home` **NEVER REFORMAT!!** on spin disk.
* `/var` Logs or DB, on spin disk
* `/tmp`, ~10GB, on spin disk
* `swap space`, 64GB, same as memory, on spin disk

## Recovery

If system cannot boot up, start with recovery mode. Some kernel versions have problem enabling the keyboard so choose one that works.

Once in recovery mode, chose **Enable Network**, this would remove the drives with write mode, which allows `root` to make changes.

Once getting out of enable network, go to `root` console. Here you can edit grub or linux kernel boot options. Or, even get rid of packages that caused problems.

For updating GRUB:

    sudo nano /etc/default/grub
    GRUB_CMDLINE_LINUX_DEFAULT="quiet splash foo=bar"
    sudo update-grub


## Random Crashes Aug 2017

Recently I've had lots of random feeze-and-crash events with Ubuntu 16.04 LTS, kernel version 4.10.0.33-generic. Nvidia driver version 381.22. Updated to deriver 384.98 but still seeing this problem.

2017-09-01: added kernel option in this [post](https://askubuntu.com/questions/761706/ubuntu-15-10-and-16-04-keep-freezing-randomly) in GRUB.

    # GRUB_CMDLINE_LINUX_DEFAULT="quiet splash pcie_aspm=off intel_idle.max_cstate=1"
    # Nov 2017 - trying with intel_idle.max_cstate=1 as posts say it's very power wasting.
    GRUB_CMDLINE_LINUX_DEFAULT="quiet splash pcie_aspm=off"

## Logs

Some logs can be cached before it's written. Do the following to turn off cache so events are logged immediately.

```
sudo vi /etc/rsyslog.d/50-default.conf
```

Find the logs files such as `-/var/log/syslog`, remove the `-` at the beginning, which indicates that events would be cached first. Reboot.

## Ethernet

Check network interface with `ifconfig`. In terminal, enable interface with, provided that `eno1` is one of the interfaces:

    sudo ip link set down eno1
    sudo ip link set up eno1

Follow [this](https://askubuntu.com/questions/4901/network-not-starting-up-on-boot) to enable ethernet at boot.

    sudo vi /etc/network/interfaces

Add the following if no present:

    auto eno1
    iface eth0 inet dhcp

## Turn off Power Management for WiFi

[here](https://itechscotland.wordpress.com/2011/09/25/how-to-permanently-turn-off-wi-fi-power-management-in-ubuntu/) and [here](http://seperohacker.blogspot.co.uk/2015/09/turning-off-wifi-power-management.html) for maximizing WiFi power. Also [here](https://unix.stackexchange.com/questions/269661/how-to-turn-off-wireless-power-management-permanently) for a more recent solution.

Run `iwconfig` in terminal to see if power management is turned on for WiFi.

To turn off:

    sudo iwconfig wlan0 power off

Permanent change:

    sudo nano /etc/pm/power.d/wireless

Place in file:

    #!/bin/bash
    /sbin/iwconfig wlan0 power off

Edit: `sudo nano /etc/NetworkManager/conf.d/default-wifi-powersave-on.conf`, set `wifi.powersave = 2`.


## DNS Server Update

Based on tips [here](https://unix.stackexchange.com/questions/128220/how-do-i-set-my-dns-when-resolv-conf-is-being-overwritten),
see the comment for Ubuntu 16.04 in the reply section for the answer from
`slm`.

```
sudo vi /etc/resolveconf/resolv.conf.d/head
# add the below 2 lines into the file:
# nameserver 1.1.1.1
# nameserver 1.0.0.1
sudo resolvconf -u

# confirm effect using command below, DNS server should be the new one.
nslookup google.com
```


## Update Ubuntu Packages

Also, I needed to **upgrade to the latest distribution linux kernel for the drivers to work**. For this, I followed the instructions [here](https://askubuntu.com/questions/196768/how-to-install-updates-via-command-line) to update the packages for the distrubtions. The commends needed:

    sudo apt-get update        # Fetches the list of available updates
    sudo apt-get upgrade       # Strictly upgrades the current packages
    sudo apt-get dist-upgrade  # Installs updates (new ones)

## Update VIM

This is needed to `.vimrc` setup, based on this [post](https://askubuntu.com/questions/284957/vi-getting-multiple-sorry-the-command-is-not-available-in-this-version-af/284960#284960).

```
sudo apt-get install vim-gui-common
sudo apt-get install vim-runtime
```

## Ubuntu Disable Suspend

Follow [this](https://askubuntu.com/questions/452908/how-to-disable-suspend-in-14-04) for version 14.04+.

Create a new file with the command below and with the following contents:

    vi /etc/polkit-1/localauthority/50-local.d/com.ubuntu.disable-suspend.pkla

    [Disable suspend (upower)]
    Identity=unix-user:*
    Action=org.freedesktop.upower.suspend
    ResultActive=no
    ResultInactive=no
    ResultAny=no

    [Disable suspend (logind)]
    Identity=unix-user:*
    Action=org.freedesktop.login1.suspend
    ResultActive=no
    ResultInactive=no
    ResultAny=no

    [Disable suspend when others are logged in (logind)]
    Identity=unix-user:*
    Action=org.freedesktop.login1.suspend-multiple-sessions
    ResultActive=no
    ResultInactive=no
    ResultAny=no

and then reboot.

This tells PolicyKit to automatically say "no" whenever anything asks if it's OK/possible to suspend. Logout menus will react to this by automatically removing the 'Suspend' menu choice.

I don't quite understand why, but the upower setting is needed for KDE but does not affect Unity. The login1 settings are needed for Unity but do not affect KDE.


Some older instructions for earlier version [here](https://askubuntu.com/questions/473037/how-to-permanently-disable-sleep-suspend), can be useful for turning off hibernation.

## Disable X screensavers

With Nvidia drvier 381.22 + 16.04 LTS my screen would not be turned back on after leaving the machine alone for a while. Turned off suspend but this still happens sometimes. Some argue that Xorg has its own screensaver setting, so have to turn this off also.

Based on this [post](https://askubuntu.com/questions/878398/ubuntu-16-04-keeps-turning-off-the-screen-after-10-minutes)

Edit `~/.xsessionrc` or `~/.xsession`, add line below:

    xset s 0 s blank
    xset s off # no screen saver
    xset s noblank # no blank screen

This [site](https://wiki.archlinux.org/index.php/Display_Power_Management_Signaling) and some other posts pointed to `xset s off` for turning off screen saver.



Useful [post](https://unix.stackexchange.com/questions/281858/difference-between-xinitrc-xsession-and-xsessionrc) on the difference between `.xsession` and `.xsessionrc`

Note: it turns out this might be a bug in the kernel. See this [link](https://askubuntu.com/questions/929479/black-screen-after-screensaver-blanking)

## Mount Apple Time Capsule

[Here](https://askubuntu.com/questions/524328/ubuntu-14-04-how-to-connect-to-apples-time-capsule), and [here](https://ineed.coffee/418/how-to-automatically-mount-and-umount-apple-time-capsule-on-linux/)

Install packages:

    sudo apt-get install cifs-utils
    sudo apt-get install smbclient

Update to scripts:

    TIMECAPSULE_IP="192.168.1.64"
    TIMECAPSULE_VOLUME="/Data"
    TIMECAPSULE_PASSWORD=password here

## Mount Options - Read Only

Use `Disks` application to change Auto mount options.

1. Find the partition to change
2. Disable auto mount
3. Change option to add `ro` for read-only.

## Ubuntu Package Installation

Refer to [here](https://askubuntu.com/questions/40779/how-do-i-install-a-deb-file-via-the-command-line) for installing packages.

To install a `.deb` package:

    sudo dpkg -i DEB_PACKAGE

To reinstall a package:

    sudo dpkg -i --reinstall DEB_PACKAGE

To resolve package dependency issues:

    sudo apt-get install -f

To remove a package:

    sudo dpkg -r PACKAGE_NAME

## Useful Ubuntu Packages

### Disk IO

    sudo apt-get install sysstat

Then use `iostat -d 30 /dev/sda5` to monitor `/dev/sda5` with refresh every 30 seconds.

### Etc

    sudo apt-get install curl
    sudo apt-get install git

    # some packages needed for R packages
    sudo apt-get install gfortran
    sudo apt-get install zlib1g-dev
    sudo apt-get install libssl-dev
    sudo apt-get install libcurl4-openssl-dev

# Font, Icon Sizes

In Screen Display, there is a scaler to adjust toolbar, font sizes. Set this
to 1.12.

##Chrome Menu Bar Size

Follow instructions on this [page](https://superuser.com/questions/1116767/chrome-ui-size-zoom-levels-in-ubuntu-16-04)

    sudo gedit /usr/share/applications/google-chrome.desktop

Find the line:

    Exec=/usr/bin/google-chrome-stable %U

Change it to:

    Exec=/usr/bin/google-chrome-stable --force-device-scale-factor=1.1 %U


## Icon Size

Based on post [here](https://askubuntu.com/questions/840613/how-to-change-desktop-icon-size-in-ubuntu-16-10)

From the launcher, type "Nautilus", which will give you the file explorer
under the confusing title "File". Then, go to Edit -> Preferences. (Do not go
to File -> Properties!) "Icon View Defaults" and "List View Defaults" are
there; use them.

## Change Mouse Cursor Size, Font Size

```
sudo apt-get install unity-tweak-tool
```
Then go to the tool, under Appearance > Cursors, change size.

Go to Appearance -> Fonts, change size.

## Unity 8

However, it is not supported by Nvidia so not working at thie time of writing... [2017-08-31]

    sudo apt-get update && sudo apt-get dist-upgrade
    sudo apt-get install unity8-desktop-session-mir

## Change Hostname

Edit the two files below, replance hostname with new name, reboot. [Source](http://ubuntuhandbook.org/index.php/2016/06/change-hostname-ubuntu-16-04-without-restart/)

    sudo nano /etc/hostname
    sudo nano /etc/hosts

## SSH on Ubuntu

Install and start the service on Server (Ubuntu):

    sudo apt-get install openssh-server
    sudo service ssh status

Set up SSH keys following this [post](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2) on client (e.g. Macbook) (passphrase is optional):

    ssh-keygen -t rsa
    # replace user@123.45.56.78 with your server user and ip, use ifconfig to find out
    ssh-copy-id user@123.45.56.78

Login to server:

    ssh user@host

Run notebooks via SSH tunneling on Client (e.g. Macbook):

    # Replace user@host with your server user and ip.
    ssh -N -f -L localhost:8888:localhost:8888 user@host

Ubuntu 16.04 LTS by default turns off root login with password. The instructions in the page used `PermitRootLogin without-password`. Ubuntu uses `prohibit-password` as a less ambiguous synonym to `without-password`.

# Nvidia Driver / GeForce 1080 Ti Founders


## Best Way to Install Cuda + Nvidia Drivers

Follow the instructions [here](http://blog.nelsonliu.me/2017/04/29/installing-and-updating-gtx-1080-ti-cuda-drivers-on-ubuntu/)

### Nvidia Driver

Use [Ubuntu Graphics Drivers PPA](https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa). Latest v381.22 is available as of 2017-06-15.

    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt-get update
    sudo apt-get install nvidia-381
    sudo reboot

If this changes the desktop environment, it is because ownership/permission issues in `HOME/.config/`. Runing `chown` and `chgrp -R` would fix it.

To remove Nvidia derivers:

```
sudo apt-get purge nvidia*
```

Command line tools `ubuntu-drivers`:

```
ubuntu-drivers list
ubuntu-drivers drives

# with drivers listed we can install them as below
# sudo apt-get nvidia-387
```

**Note**: As of Nvidia driver version 384.69 that came out on 2017-08-22, `pytorch` correctly sees cuda with `torch.cuda.is_availalbe()` after installing using Nvidia's scripts. This is done on a fresh Ubuntu install. However, after updating Ubuntu packages and a reboot, system failed to boot into X. This [post](http://tipsonubuntu.com/2017/08/23/nvidia-384-69-released-new-gpu-support-fixes/) is very useful in showing how to roll back and uninstall Nvidia official drivers.

If you get a black screen after installing the derivers, either remove nvidia drivers in recovery mode, or set `nomodeset` in GRUB:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nomodeset pcie_aspm=off intel_idle.max_cstate=1"
```

**Currently the best way is still to use `ppa:graphics-drivers/ppa` to install version 381.**

Run `nvidia-smi` to confirm driver version is the latest.

Compile `/usr/local/cuda/samples/1_Utilities/deviceQuery` to test.

### Cuda

**Install with runfile, opt out of Nvidia driver installation as it is an older version.**

Choose Y for creating symbolic link for `/usr/local/cuda`.

**Don't forget to set:** `umask 022`

* Add `/usr/local/cuda/bin` to `PATH`
* Add `/usr/local/cuda/lib64` to `LD_LIBRARY_PATH`

### cuDNN 7.0 for Cuda 9.0

Download page [here](https://developer.nvidia.com/rdp/cudnn-download), download both runtime and dev libraries. Perhaps docs if you want to run the tests.

Installation guide [here](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

Multiple versions of `cuda` can be installed at the some time. A symbolic link
is created (`/usr/local/cuda`) to link to a `cuda` version chosen.

### PCIe Bus ERROR

This was written for Nvidia drvier version 381.22.

This seems to be a problem for Asus, X99 chipset & PLX switches issue, according to this [post](https://forums.geforce.com/default/topic/957456/geforce-drivers/gtx-1080-throwing-bad-tlp-pcie-bus-errors/2/).

I used one of the suggestions by adding this kernel parameters: `pcie_aspm=off`. This seems to have worked for me so far, no more error messages. **Note: this is causing problem at start up for Nvidia driver v 384.69.**

Follow this [page](https://askubuntu.com/questions/19486/how-do-i-add-a-kernel-boot-parameter) on how to add kernel boot options.

## Previous Failed Attempts

A few issues I encountered here:

### Driver Installation Issues, version 381.22

Had problem with this installation, potentially due to the fact that it was a reinstall. In a reinstall, some GL libs are **skipped** and not installed. Solution was to add manual switches to force them to be reinstalled.

However, I still cannot get the `drm` tools to work.

The parameters I used:

    sudo ./NVIDIA-Linux-x86_64-381.22.run -install-libglvnd -glvnd-glx-client -glvnd-egl-client -no-drm

## Others

### Disable Secure Boot

Some posts suggested disabling secure book in BIOS helped with the `drm` issue, but this did not work for me.



## Software Updates

### Conda

Edit `.condarc` to reorder channel priorities.

### Python packages

    conda install pytorch torchvision cuda91 -c pytorch
    conda install hdbscan tabulate joblib holoviews
    conda install seaborn tqdm watermark xarray datashader

    conda install -c districtdatalabs yellowbrick
    # conda install -c bokeh datashader

    # for difficult to build cartopy and pynio.
    conda install -c conda-forge cartopy pynio

    pip install hypertools janitor

    pip install git+https://github.com/bashtage/arch.git
    pip install --upgrade google-api-python-client

    # 2to3 converter & AutoPEP 8 in jupyter
    conda install jupyter_contrib_nbextensions autopep8

    # enable config page
    jupyter nbextensions_configurator enable --user

    # filterpy
    git clone https://github.com/rlabbe/filterpy.git
    cd filterpy
    python setup.py install

2018-03-04: `tensorflow 1.6` currently uses cuda 9.0, whereas `pytorch 0.3.1`
uses cuda 9.1.

`keras`: install from github source, will also need to install `pyyaml`:

```
conda install pyyaml
```


### Jupyter Notebook

#### Password

To setup a password for jupyter server, see [here](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html)


#### Matplotlib `figure.figsize`

Jupyter notebook uses iPython for python kernels. Therefore we need to change:
```
~/.ipython/profile_default/ipython_kernel_config.py
```

This is created through `ipython profile create`. Insert the following at the end of the file:

```
c.InlineBackend.rc = {'figure.figsize': (12, 4)}
```


#### Cell Width

Cell width can be changed by editing `./jupyter/custom/custom.css` file.
File content should be:

```
.container { width:90% !important; }
```


#### Crontab

Useful tip from this [link](https://stackoverflow.com/questions/2229825/where-can-i-set-environment-variables-that-crontab-will-use)

Instead of calling `jupyter notebook` directly in the crontab scripts, call a script that sets up the needed environment variables, such as `PYTHONPATH`, to start `jupyter notebook` server. See [this script](./cron_jupyter.sh).

`Crontab` startup logs can be found in `/var/log/syslog`

Step:
    1. sudo crontab -e
    2. insert line: `@reboot \home\zwl\.cron_jupyter.sh`
    3. Save and exit

#### Extension Widgets

[`jupyter-contrib-nbextensions`](http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html#enabling-disabling-extensions) has a `2to3 converter` to convert Python 2.x notebooks to 3.x.

```
conda install jupyter_contrib_nbextensions
```

#### Showing Different Conda envs

Anaconda stopped shipping this support by default. See [here](https://github.com/jupyter/jupyter/issues/245). As suggested, the instructions to manually include these are [here](http://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments).

Essentiall, first activate the new environment, run the following:

```
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

## RStudio / MRO

### MRO

Installs to `/opt/microsoft/ropen`, create symlink to current version.

`3.5.0` had issue with `libpng12.so.0`, fixed following instructions [here](https://github.com/Microsoft/microsoft-r-open/issues/34)

Run the following commands:

```
cd /usr/lib/x86_64-linux-gnu
sudo ln -s libpng16.so.16.34.0 libpng12.so.0
```

### RStudio

[Download page](https://www.rstudio.com/products/rstudio/download/)

Download the tarball and unpack into `/usr/local/rstudio`, create symbolic link
to the latest version.

On `18.04` there are some lib problems. So the tarball method won't work.
Install with `deb` package for `16.04` still works.


Install package with the following, found [here](https://support.rstudio.com/hc/en-us/community/posts/209074748-Installing-RStudio-Desktop-on-Ubuntu-16-04-LTS).

    sudo dpkg -i downloaded_file.deb
    sudo apt-get -f install

To remove:

    sudo dpkg --purge rstudio

### Rethinking / Rstan

Create `.Rprofile` file and write the line below for MRO.

    options(repos = c(CRAN = "https://cran.revolutionanalytics.com"))

Need to install a few libraries.

    sudo apt-get install libssl-dev
    sudo apt-get install r-base

    # needed for some packages
    # sudo apt-get install libgfortran4

    # after upgrading to gcc 7.3 from jonathonf/gcc ppa, gfortran-7 is now
    # also available, so there is no need to perform the below from ubuntu's
    # ppa. Just get both gcc and gfortran-7 from jonathonf/gcc PPA.
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test
    sudo apt-get install gfortran-7

`rstan` Installation [here](https://github.com/stan-dev/rstan/wiki/Installing-RStan-on-Mac-or-Linux), make sure `.R/Makevars` file is set
correctly.


Then run in `R`:

    install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies=TRUE)

If SSL authentication issues pop up, need to installed some other packages
with `apt-get` first. See section above on useful packages. See this
[issue](https://github.com/hadley/devtools/issues/1079), then run:

    install.packages(c("curl", "httr"))

Installing `Rethinking`:

    install.packages(c("coda","mvtnorm","devtools","loo"))
    library(devtools)
    devtools::install_github("rmcelreath/rethinking")


Might need to do a local install:

    git clone https://github.com/rmcelreath/rethinking.git
    library(devtools)
    devtools::install('/home/zwl/git/rethinking/')

Other popular packages to install & summary of above:

```
install.packages(c("curl", "httr"))
install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies=TRUE)
# rethinking
install.packages(c("coda","mvtnorm","devtools","loo"))
library(devtools)
devtools::install_github("rmcelreath/rethinking")
# others
install.packages(c('ISLR', 'caret', 'MASS', 'car', 'glmnet', 'gvlma'))
install.packages(c('lubridate', 'xts', 'googleVis', 'leaps', 'bootstrap'))
install.packages(c('ggplot2', 'dplyr', 'stringr', 'tidyr'))
```

To install `R` package `edgarWebR`, need to install `libxml2-dev`:

    sudo apt-get install libxml2-dev

### TA-Lib

Need to install `TA-Lib` c libraries first. Follow instructions on [github](https://github.com/mrjbq7/ta-lib).

Download library [here](http://www.ta-lib.org/hdr_dw.html)

    untar and cd
    ./configure --prefix=/usr
    make
    umask 022
    sudo make install

    pip install TA-Lib

## Sublime Text 3

User settings / key bindings can be found in Gmail. Labelled as `Code`.

Instructions [here](https://realpython.com/blog/python/setting-up-sublime-text-3-for-full-stack-python-development/)

Packages installed:

    Anaconda
    AutoDocstring
    GitGutter
    Side bar
    Markdown Preview
    Package Control
    Python PEP8 Autoformat
    SublimeLinter

### Sidebar / Tab Font size

Install package: `PackageResourceViewer`

Run `PackageResourceViewer`, edit `Adaptive.sublime-theme`.

Code needed:

```
    {
        "class": "sidebar_label",
        "fg": ["background", 255, 255, 255, 0.8],
        "font.size": 18
    },
    {
        "class": "tab_label",
        "fg": [152, 152, 152],
        "font.size": 16,
        "shadow_color": [0, 0, 0, 0.25],
        "shadow_offset": [0, -1]
    },

```


## GCC v7

Avaliable [here](https://launchpad.net/~ubuntu-toolchain-r/+archive/ubuntu/test)

```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-7
```

For **gcc-7.3** use [jonathanf/gcc](https://launchpad.net/~jonathonf/+archive/ubuntu/gcc)

```
sudo add-apt-repository ppa:jonathonf/gcc
sudo apt-get update
sudo apt-get install gcc-7
sudo apt-get install gfortran-7
```

### Change GCC/gfortran default version

`update-alternatives`: lower priority number has higher priority.

```
sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-7 50
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 50
sudo update-alternatives --config gcc
```


## Manage PPA

Go to Software & Updates -> Other Software, to disable a particular PPA

Simple way:

```
sudo add-apt-repository --remove ppa:whatever/ppa
```

Or, see this [post](https://askubuntu.com/questions/307/how-can-ppas-be-removed)

```
sudo apt-get install ppa-purge
# this would downgrade the software installed from tis PPA
sudo ppa-purge ppa:whatever/ppa
```


## APT Dependencies

This [post](https://askubuntu.com/questions/140246/how-do-i-resolve-unmet-dependencies-after-adding-a-ppa) is great for trying to
debug `apt` dependency issues.

Very often dependency issues can be resolved by running the below. This
worked for installing `openjdk-8-jdk` for compiling `tensorflow` from source.

```
sudo apt-get install -f
```

## Kernel Update

Based on post [here](https://www.howtoforge.com/tutorial/how-to-upgrade-linux-kernel-in-ubuntu-1604-server/#step-install-new-kernel-version) and [here](http://ubuntuhandbook.org/index.php/2017/09/install-linux-kernel-4-13-ubuntu-16-04-higher/). Instructions below worked for kernel 4.13.

Running kernel 4.14 from 2017-01-02, let's see if the kernel panic events come back...

First, update all packages. Then run the following

```
mkdir kernels

cd kernels

wget http://kernel.ubuntu.com/~kernel-ppa/mainline/v4.13/linux-headers-4.13.0-041300_4.13.0-041300.201709031731_all.deb

wget http://kernel.ubuntu.com/~kernel-ppa/mainline/v4.13/linux-headers-4.13.0-041300-generic_4.13.0-041300.201709031731_amd64.deb

wget http://kernel.ubuntu.com/~kernel-ppa/mainline/v4.13/linux-image-4.13.0-041300-generic_4.13.0-041300.201709031731_amd64.deb

sudo dpkg -i *.deb

sudo update-grub

# sudo reboot

```

To check current kernel version: `uname -msr`

When trying 4.14.10, I noticed that `libelf-dev` is required for some drivers. Also this version requires nvidia-ppa v387.34 to work.

```
sudo apt-get install libelf-dev
```


### Remove old kernel

Run script in `~/bin/list_kernels.sh`. Find a kernel with `ii` tag at the beginning of the line, these are safe to remove. Also make sure to **never** remove a kernel currently in use!

* rc: means it has already been removed.
* ii: means installed, eligible for removal.
* iU: DON’T REMOVE. It means not installed, but queued for install in apt.

```
sudo dpkg --purge kernel-image-name
```

Remove any failed dependencies with `sudo dpkg --purge package_name`.

## Chinese Input

    sudo apt-get install sogoupinyin
    sudo reboot


# Citrix Receiver

1. Download deb package from Citrix website
2. use `gdebi filename.deb` to install the Citrix app, [here](https://askubuntu.com/questions/40723/how-do-i-install-citrix-receiver) the
package will also show up in `sudo apt list --install | grep icaclient`.
3. Add CA certificates, [here](https://help.ubuntu.com/community/CitrixICAClientHowTo):

```
sudo ln -s /usr/share/ca-certificates/mozilla/* /opt/Citrix/ICAClient/keystore/cacerts/
sudo /usr/bin/c_rehash /opt/Citrix/ICAClient/keystore/cacerts/
```

For Ubuntu 16.04, Citrix would be installed in `/opt/Citrix`.

For configuration, run `/opt/Citrix/ICAClient/util/configmgr`.

To uninstall:

```
sudo apt-get remove icaclient
```

# Ubuntu 18.04

[Ubuntu UEFI Guide](https://help.ubuntu.com/community/UEFI), shows that there
are different screens when booting with installation usb in UEFI or legacy
BIOS modes.

Some notes on Nvidia/Cuda related installations [here](https://askubuntu.com/questions/1033489/the-easy-way-install-nvidia-drivers-cuda-cudnn-and-tensorflow-gpu-on-ubuntu-1)

```
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
nvidia-smi

# below installs cuda 9.1, or manually install instead.
sudo apt install nvidia-cuda-toolkit
```

To fix the grub issue, boot into ubuntu installabtion in non-UEFI mode,
because my Windows 10 is installed with MBR not GPT partitions.

## Packages

See `scripts` folder for more details.

## Settings

**Workspaces**

Install `Gnome Tweaks`, go to Workspaces setting.

# Fix Windows Boot

Based on this [post](https://www.quora.com/How-do-I-boot-into-Windows-10-from-Grub-Rescue-Mode), worked on my machine.

Boot with Windows 10 CD, goto Repair your computer -> Troubleshooting ->
Command prompt, run:

```
Bootrec /fixmbr
```

## Convert from MBR to GPT

Use this tool [mbr2gpt](https://docs.microsoft.com/en-us/windows/deployment/mbr-to-gpt).

Intel also has a [page](https://www.intel.com/content/www/us/en/support/articles/000024558/memory-and-storage/intel-optane-memory.html) on this.

While running Windows 10, need to use `/allowFullOS` flag.
```
mbr2gpt /validate /allowFullOS
mbr2gpt /convert /allowFullOS
```

After conversion, go to motherboard bios and change to UEFI boot. Also install
Ubuntu with UEFI boot. This should fix the issue.

To install Ubuntu:

Goto Windows Settings -> Update & Securities -> Recovery -> Advanced Options
-> Reboot
