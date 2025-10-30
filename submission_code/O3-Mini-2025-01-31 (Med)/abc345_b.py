def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    X = int(input_data[0])
    # We can compute the ceiling of X/10 using integer arithmetic.
    # The expression -(-X // 10) gives the ceiling division result when dividing X by 10.
    result = -(-X // 10)
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()