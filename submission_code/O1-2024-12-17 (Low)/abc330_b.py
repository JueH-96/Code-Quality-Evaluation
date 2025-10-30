def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, L, R = map(int, data[:3])
    A = list(map(int, data[3:] ))
    
    result = []
    for val in A:
        if val < L:
            result.append(L)
        elif val > R:
            result.append(R)
        else:
            result.append(val)
    print(" ".join(map(str, result)))

# DO NOT FORGET TO CALL MAIN
main()