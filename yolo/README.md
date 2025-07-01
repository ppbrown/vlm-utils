# YOLO image taggger suppport

YOLO is the real-world image equivalent of WD tagger.
For maximum speed, you want the RT compiled version.

./yolo-install.sh

will take care of compling that for you.

After you have done that you will be able to run

./yolo-dir.sh  /some/image/directory


and it will create .txt files to match all your img files.

WARNING: If it cannot detect any recognizable objects, it will not
create a .txt file at all


