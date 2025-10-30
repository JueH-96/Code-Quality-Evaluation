from itertools import combinations

def main():
    import sys
    K = int(sys.stdin.read())
    numbers = []
    for size in range(1, 11):
        for comb in combinations(range(0,10), size):
            if size ==1 and comb[0]==0:
                continue
            digits = sorted(comb, reverse=True)
            number = int(''.join(map(str, digits)))
            numbers.append(number)
    numbers.sort()
    print(numbers[K-1])

if __name__ == "__main__":
    main()