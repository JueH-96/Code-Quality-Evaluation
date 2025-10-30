class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        visited_integers = []
        result = []
        for i in range(len(words)):
            word = words[i]
            if word == "prev":
                k = 0
                for j in range(i, -1, -1):
                    if words[j] == "prev":
                        k += 1
                    else:
                        break
                nums_reverse = visited_integers[::-1]
                if k > len(visited_integers):
                    last_visited_integer = -1
                else:
                    last_visited_integer = nums_reverse[k-1] if k > 0 else -1
                result.append(last_visited_integer)
            else:
                visited_integers.append(int(word))
        return result