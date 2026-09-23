tu_tich_cuc = ["ngon", "tốt", "thích", "tuyệt", "vui"] 
tu_tieu_cuc = ["dở", "tệ", "ghét", "chán", "buồn"]

cau_test = "Món ăn này rất ngon và tôi rất thích"
s = cau_test.split()
diem_pos=0
diem_neg=0
for word in range(len(s)):
    if word in tu_tich_cuc:
        diem_pos += 1
    elif word in tu_tieu_cuc:
        diem_neg +=1
if diem_pos > diem_neg:
    print("tich cuc")
elif diem_neg > diem_pos:
    print("tieu cuc")
elif diem_neg == diem_pos:
    print("trung tinh")