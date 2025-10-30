# YOUR CODE HERE
def can_color_all(n, k):
    if n % 2 == 0:
        return k % 2 == 0
    else:
        return True

def main():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        print("Yes" if can_color_all(n, k) else "No")

if __name__ == "__main__":
    main()