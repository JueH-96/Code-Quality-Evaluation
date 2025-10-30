def main():
    n, l, r = map(int, input().split())
    sequence = list(map(int, input().split()))
    
    result = []
    for a in sequence:
        # If a is less than L, then L is the closest value in range
        # If a is greater than R, then R is the closest value in range
        # Otherwise, a itself is the closest value in range
        if a <= l:
            result.append(l)
        elif a >= r:
            result.append(r)
        else:
            result.append(a)
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()