#!/bin/env python
"""
Util to compare images in checkpoint save subdirs.
Give it two directories,and it will attempt to display sequential images
side by side.
This makes it easier to check if current run is better or worse than previous one
"""

import sys, os, argparse
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QHBoxLayout, QMessageBox
)
from PySide6.QtGui import QPixmap, QKeyEvent
from PySide6.QtCore import Qt

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

def find_images_recursive(folder):
    image_files = []
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(VALID_EXTENSIONS):
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, folder)
                image_files.append((rel_path, full_path))
    image_files.sort()
    return image_files

class ImageCompareViewer(QWidget):
    def __init__(self, folder1, folder2):
        super().__init__()
        self.setWindowTitle("Recursive Image Comparator")
        self.resize(1040, 520)

        self.folder1 = folder1
        self.folder2 = folder2

        images1 = find_images_recursive(folder1)
        images2 = find_images_recursive(folder2)

        # Match by relative path
        common_keys = sorted(set(k for k, _ in images1) & set(k for k, _ in images2))
        self.pairs = [(dict(images1)[k], dict(images2)[k]) for k in common_keys]
        self.max_index = len(self.pairs)
        self.index = 0

        self.label1 = QLabel("Folder 1")
        self.label2 = QLabel("Folder 2")
        self.label1.setAlignment(Qt.AlignCenter)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label1.setScaledContents(True)
        self.label2.setScaledContents(True)

        layout = QHBoxLayout()
        layout.addWidget(self.label1)
        layout.addWidget(self.label2)
        self.setLayout(layout)

        if not self.pairs:
            QMessageBox.critical(self, "No matches", "No matching image filenames found.")
            sys.exit(1)

        self.update_images()

    def update_images(self):
        def prepare_pixmap(path):
            pixmap = QPixmap(path)
            if pixmap.isNull():
                return QPixmap(512, 512)  # fallback blank
            pixmap = pixmap.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # center-crop to 512×512
            x = max(0, (pixmap.width() - 512) // 2)
            y = max(0, (pixmap.height() - 512) // 2)
            return pixmap.copy(x, y, 512, 512)

        if 0 <= self.index < self.max_index:
            img1, img2 = self.pairs[self.index]
            self.label1.setPixmap(prepare_pixmap(img1))
            self.label2.setPixmap(prepare_pixmap(img2))
            basename = os.path.basename(self.pairs[self.index][0])
            self.setWindowTitle(f"[{self.index+1}/{self.max_index}] {basename}")
        else:
            QMessageBox.warning(self, "Out of range", "No more images.")


    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_Right or key == Qt.Key_Space:
            self.index = min(self.index + 1, self.max_index - 1)
            self.update_images()
        elif key == Qt.Key_Left or key == Qt.Key_Backspace:
            self.index = max(self.index - 1, 0)
            self.update_images()
        elif key == Qt.Key_Escape or key == Qt.Key_Q:
            self.close()

def main():
    parser = argparse.ArgumentParser(description="Recursively compare image folders side-by-side.")
    parser.add_argument("folder1", help="First folder")
    parser.add_argument("folder2", help="Second folder")
    args = parser.parse_args()

    if not os.path.isdir(args.folder1) or not os.path.isdir(args.folder2):
        print("Error: One or both paths are not valid directories.")
        sys.exit(1)

    app = QApplication(sys.argv)
    viewer = ImageCompareViewer(args.folder1, args.folder2)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
