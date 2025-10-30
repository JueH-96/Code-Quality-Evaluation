def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    X = int(data[0])
    
    # Since we need ceil(X / 10) without floating point arithmetic, we do the following.
    if X >= 0:
        result = (X + 9) // 10
    else:
        # For negative X, Python's // performs floor division. If X is not exactly divisible,
        # then floor division will be lower than the ceiling. So we adjust by adding 1.
        if X % 10 == 0:
            result = X // 10
        else:
            result = X // 10 + 1
    
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()