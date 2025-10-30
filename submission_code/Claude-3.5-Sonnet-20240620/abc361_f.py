# YOUR CODE HERE
def count_expressible(N):
    seen = set()
    for a in range(2, int(N**0.5) + 1):
        b = 2
        while True:
            x = a**b
            if x > N:
                break
            seen.add(x)
            b += 1
    return len(seen) + 1  # +1 for 1, which is always expressible as 1^n

N = int(input())
print(count_expressible(N))