import sys

def find_integer(B):
    i = 1
    while i**i <= B:
        if i**i == B:
            return i
        i += 1
    return -1

B = int(sys.stdin.readline().strip())
print(find_integer(B))