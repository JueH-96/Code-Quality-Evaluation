def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    A, B, C = map(int, data)
    
    # Check partition into two groups (one element and two elements)
    if A == B + C or B == A + C or C == A + B:
        print("Yes")
        return

    # Check partition into three groups (only possible if all are equal)
    if A == B == C:
        print("Yes")
        return

    print("No")
    
if __name__ == '__main__':
    main()