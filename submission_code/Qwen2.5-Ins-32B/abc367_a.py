import sys

def can_shout_love(A, B, C):
    if B < C:
        return B <= A < C
    else:
        return A >= C or A < B

def main():
    A, B, C = map(int, input().split())
    if can_shout_love(A, B, C):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()