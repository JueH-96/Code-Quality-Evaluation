from math import comb

def count_321_like_numbers(d, k):
    if d == 1:
        return 9
    if k == 1:
        return 1
    if k == 0:
        return 0
    total = 0
    for i in range(1, 10):
        total += comb(9 - i, d - 1)
        if total >= k:
            return i * 10**(d - 1) + count_321_like_numbers(d - 1, k - (total - comb(9 - i, d - 1)))
    return 0

def main():
    K = int(input())
    d = 1
    while True:
        if count_321_like_numbers(d, K) > 0:
            break
        d += 1
    print(count_321_like_numbers(d, K))

if __name__ == "__main__":
    main()