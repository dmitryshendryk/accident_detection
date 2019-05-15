cd install
chmod x+a download_models.sh
./download_models.sh

cd ..
sudo docker build -t accident_detection  .