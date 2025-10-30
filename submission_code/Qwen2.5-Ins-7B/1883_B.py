# YOUR CODE HERE
t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input()
    char_count = [0] * 26
    for c in s:
        char_count[ord(c) - ord('a')] += 1
    odd_count = sum(count % 2 for count in char_count)
    if odd_count - k <= 1 and (n - k) % 2 == 0:
        print("YES")
    else:
        print("NO")