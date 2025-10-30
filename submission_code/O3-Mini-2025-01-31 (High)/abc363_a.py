def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    R = int(input_data[0])
    
    if R < 100:
        result = 100 - R
    elif R < 200:
        result = 200 - R
    else:
        result = 300 - R

    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()