import imghdr
import os
import tensorflow as tf

# path = r"D:\MyFiles\ResearchSubject\3dmax\doorModel_datasets"
# list = os.listdir(path)
# for listi in list:
#     imgPath = os.path.join(path, listi)
#     for img in os.listdir(imgPath):
#         img1 = os.path.join(imgPath, img)
#         if imghdr.what(img1):
#             print("正确")
#         else:
#             print("错误")
#
# import os
#
# num_skipped = 0
# for folder_name in ("Cat", "Dog"):
#     folder_path = os.path.join("PetImages", folder_name)
#     for fname in os.listdir(folder_path):
#         fpath = os.path.join(folder_path, fname)
#         try:
#             fobj = open(fpath, "rb")
#             is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
#         finally:
#             fobj.close()
#
#         if not is_jfif:
#             num_skipped += 1
#             # Delete corrupted image
#             os.remove(fpath)
#
# print("Deleted %d images" % num_skipped)

path = r"D:\MyFiles\ResearchSubject\3dmax\doorModel_datasets"
list = os.listdir(path)
# if os.path.exists(path+"\\desktop.ini"):
#     os.remove(path+"\\desktop.ini")
print(list)
print(len(list))
for listi in list:
    imgPath = os.path.join(path, listi)
    for img in os.listdir(imgPath):
        img1 = os.path.join(imgPath, img)
        fobj = open(img1, "rb")
        is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        if is_jfif == False:
            print("？？")

def is_valid_jpg(jpg_file):
    with open(jpg_file, 'rb') as f:
        f.seek(-2, 2)
        buf = f.read()
        f.close()
        return buf ==  b'\xff\xd9'  # 判定jpg是否包含结束字段

path = r"D:\MyFiles\ResearchSubject\3dmax\doorModel_datasets"
list = os.listdir(path)
for listi in list:
    imgPath = os.path.join(path, listi)
    for img in os.listdir(imgPath):
        img1 = os.path.join(imgPath, img)
        if is_valid_jpg(img1)  == False:
            print(img1)

