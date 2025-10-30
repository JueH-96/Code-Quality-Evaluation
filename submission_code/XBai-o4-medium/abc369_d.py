import sys

def main():
    n = int(sys.stdin.readline())
    a_list = list(map(int, sys.stdin.readline().split()))
    even = 0
    odd = -float('inf')
    for a in a_list:
        new_even = max(even, odd + 2 * a)
        new_odd = max(odd, even + a)
        even, odd = new_even, new_odd
    print(max(even, odd))

if __name__ == "__main__":
    main()