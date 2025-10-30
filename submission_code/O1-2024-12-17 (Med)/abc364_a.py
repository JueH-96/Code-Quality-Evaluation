def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    dishes = input_data[1:]  # list of "sweet" or "salty"
    
    # Check for consecutive "sweet"
    for i in range(N - 1):
        if dishes[i] == "sweet" and dishes[i + 1] == "sweet":
            # If the second sweet dish is not the last dish, he cannot finish
            if i + 1 != N - 1:
                print("No")
                return
            else:
                # If it's the last dish, he's finished them all
                print("Yes")
                return
    
    # If no consecutive sweet found, he can finish all
    print("Yes")

# Do not forget to call main()
main()