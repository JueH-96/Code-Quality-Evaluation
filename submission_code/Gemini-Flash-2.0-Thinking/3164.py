class Solution:
    def lastVisitedIntegers(self, words: List[str]) -> List[int]:
        visited_integers = []
        result = []

        for i in range(len(words)):
            if words[i] == "prev":
                k = 0
                for j in range(i, -1, -1):
                    if words[j] == "prev":
                        k += 1
                    else:
                        break

                if k <= len(visited_integers):
                    reversed_visited_integers = visited_integers[::-1]
                    last_visited = reversed_visited_integers[k - 1]
                    result.append(last_visited)
                else:
                    result.append(-1)
            else:
                visited_integers.append(int(words[i]))

        return result