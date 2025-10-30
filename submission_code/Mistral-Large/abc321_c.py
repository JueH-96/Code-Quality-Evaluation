import itertools

def is_321_like(number):
    str_num = str(number)
    return all(str_num[i] > str_num[i+1] for i in range(len(str_num)-1))

def find_kth_321_like_number(k):
    count = 0
    num = 1
    while count < k:
        if is_321_like(num):
            count += 1
        num += 1
    return num - 1

if __name__ == "__main__":
    import sys
    input = sys.stdin.read()
    k = int(input.strip())
    result = find_kth_321_like_number(k)
    print(result)