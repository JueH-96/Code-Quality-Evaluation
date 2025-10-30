def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    N = int(input_data)
    
    def is_326_like(num):
        # Get digits
        hundreds = num // 100
        tens = (num // 10) % 10
        ones = num % 10
        return (hundreds * tens) == ones

    # Since constraint is N >= 100 and always exists under constraints.
    for num in range(N, 1000):
        if is_326_like(num):
            print(num)
            return

if __name__ == '__main__':
    main()