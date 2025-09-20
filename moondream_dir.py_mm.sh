#!/bin/bash

if [[ ! -d "$1" ]] ; then
	echo ERROR: must give a directory name
	exit 1
fi

MYDIR=$(dirname $0)

$MYDIR/moondream_dir.py -p 'what is the medium of the image' -s moon.mm $1

echo ""
echo Now try:
echo "find $1 -name *.moon.mm|xargs egrep -l 'watercolor|painting|oil paint|etch|illustration|black and white'|sed s/moon.mm/txt/|sfeh"

