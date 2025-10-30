import sys

def kth_321_like_number(K):
    count = 0
    num = 1
    while True:
        if all(int(num[i]) > int(num[i+1]) for i in range(len(num)-1)):
            count += 1
            if count == K:
                return int(num)
        num = str(int(num) + 1)

# Read input
K = int(sys.stdin.read().strip())

# Output the result
print(kth_321_like_number(K))