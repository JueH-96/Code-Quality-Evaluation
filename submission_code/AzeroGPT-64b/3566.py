class Solution:
    def stringSequence(self, target: str) -> List[str]:
        sol = ["a"]
        convert = "abcdefghijklmnopqrstuvwxyz"
        
        for i,c in enumerate(target):
            temp = []
            counter = 0
            var = sol[-1]
            while var[-1] != c:
                var = list(var)
                index = convert.index(var[-1])
                var[-1] = convert[(index + 1)%26]
                var = "".join(var)
                temp.append(var)
                if len(sol) > 2000:
                    return sol
                counter += 1
            sol += ["a"]*counter + [j+"a" for j in temp]
        return sol