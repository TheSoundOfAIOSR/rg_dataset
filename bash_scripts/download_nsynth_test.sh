
#!/bin/bash
DOWNLOAD_PATH="http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"

read -r -p "This will download the NSynth test set (~333MB), which may take a long time.
Are you sure you want to do this? [y/N] " response
case "$response" in
    [yY][eE][sS]|[yY])
	read -p "Enter dataset path [../datasets/nsynth-test]: " dataset_path
	dataset_path=${dataset_path:-../datasets}
        mkdir -p $dataset_path
	curl -o $dataset_path/nsynth_tar $DOWNLOAD_PATH
        tar -xvf $dataset_path/nsynth_tar -C $dataset_path
	rm -f $dataset_path/nsynth_tar
	echo "Done."
        ;;
    *)
        echo "Aborting."
        ;;
esac
