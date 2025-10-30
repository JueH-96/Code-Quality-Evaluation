def main():
    t = int(input().strip())
    for _ in range(t):
        s = input().strip()
        
        # If it's already 'abc', no swaps are needed
        if s == "abc":
            print("YES")
            continue
        
        # Check if a single swap can turn s into 'abc'
        possible = False
        for i in range(3):
            for j in range(i+1, 3):
                arr = list(s)
                arr[i], arr[j] = arr[j], arr[i]
                if "".join(arr) == "abc":
                    possible = True
                    break
            if possible:
                break
        
        print("YES" if possible else "NO")

# Don't forget to call main()
if __name__ == "__main__":
    main()