sudo apt update && sudo apt-get update

echo "system updated"

wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh

bash Anaconda3-2021.11-Linux-x86_64.sh

export PATH=~/anaconda3/bin:$PATH

echo "Path is exporteds"

echo "Close the connection and reconnect"