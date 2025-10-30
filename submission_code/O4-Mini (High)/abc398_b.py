from collections import Counter

def main():
    A = list(map(int, input().split()))
    counts = Counter(A)
    
    # Try every value x for the triple
    for x, cx in counts.items():
        if cx >= 3:
            # And every different value y for the pair
            for y, cy in counts.items():
                if y != x and cy >= 2:
                    print("Yes")
                    return
    
    print("No")

if __name__ == "__main__":
    main()