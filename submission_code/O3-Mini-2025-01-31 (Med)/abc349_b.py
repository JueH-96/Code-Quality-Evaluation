def main():
    import sys
    from collections import Counter

    input_str = sys.stdin.read().strip()
    if not input_str:
        return

    freq = Counter(input_str)
    max_freq = max(freq.values())
    
    for i in range(1, max_freq + 1):
        count_i = sum(1 for v in freq.values() if v == i)
        if count_i not in (0, 2):
            print("No")
            return
            
    print("Yes")
    
if __name__ == "__main__":
    main()