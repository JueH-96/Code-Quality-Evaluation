def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    strings = input_data[1:]
    
    for i in range(N):
        for j in range(N):
            if i != j:
                concatenated = strings[i] + strings[j]
                if concatenated == concatenated[::-1]:
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()