def main():
    import sys
    input_value = sys.stdin.read().strip()
    if input_value:
        x = float(input_value)
        if x >= 38.0:
            print(1)
        elif x >= 37.5:
            print(2)
        else:
            print(3)

if __name__ == '__main__':
    main()