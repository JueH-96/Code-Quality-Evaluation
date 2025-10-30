def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # The first element is S (surname), the second element is T (first name)
    if len(input_data) < 2:
        return
    surname = input_data[0]
    result = surname + " san"
    print(result)

if __name__ == '__main__':
    main()