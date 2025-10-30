def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def get_creases(folds):
        creases = []
        def generate_creases(folds_remaining, current_creases):
            if folds_remaining == 0:
                creases.append(current_creases)
                return
            
            new_creases = current_creases + ["V"]
            
            temp = []
            for crease in reversed(current_creases):
                if crease == "V":
                    temp.append("M")
                else:
                    temp.append("V")
            
            new_creases = new_creases + temp
            
            generate_creases(folds_remaining - 1, new_creases)
        
        generate_creases(folds, [])
        return creases[0]

    creases = get_creases(100)
    num_creases = len(creases)
    
    max_f = 0
    for i in range(num_creases - a[-1]):
        f_i = 0
        for k in range(n):
            if creases[i + a[k]] == "M":
                f_i += 1
        max_f = max(max_f, f_i)
    
    print(max_f)

solve()