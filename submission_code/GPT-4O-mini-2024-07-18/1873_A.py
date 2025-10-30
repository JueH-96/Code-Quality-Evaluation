def can_form_abc(s):
    if s == "abc":
        return "YES"
    elif s in ["acb", "bac", "cba"]:
        return "YES"
    else:
        return "NO"

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    t = int(data[0])
    results = []
    
    for i in range(1, t + 1):
        results.append(can_form_abc(data[i]))
    
    print("
".join(results))

if __name__ == "__main__":
    main()