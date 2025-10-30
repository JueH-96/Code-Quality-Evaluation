def is_321_like(n):
    s = str(n)
    for i in range(len(s)-1):
        if int(s[i]) <= int(s[i+1]):
            return False
    return True

def find_kth_321_like(k):
    count = 0
    num = 1
    
    while True:
        if is_321_like(num):
            count += 1
            if count == k:
                return num
        num += 1

K = int(input())
print(find_kth_321_like(K))