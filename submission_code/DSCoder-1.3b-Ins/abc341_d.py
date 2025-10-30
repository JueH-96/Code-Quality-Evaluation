import sys

def find_kth_smallest(n, m, k):
    i = 1
    while True:
        if i % n == 0 and i % m == 0:
            k -= 1
        if k == 0:
            return i
        i += 1

def main():
    n, m, k = map(int, sys.stdin.readline().split())
    print(find_kth_smallest(n, m, k))

if __name__ == "__main__":
    main()