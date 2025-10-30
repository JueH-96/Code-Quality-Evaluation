class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited_friends = set()
        current_friend = 1
        turn_number = 1
        visited_friends.add(current_friend)
        friends = list(range(1, n + 1))
        while True:
            next_friend_index = (current_friend - 1 + turn_number * k) % n + 1
            if next_friend_index in visited_friends:
                break
            visited_friends.add(next_friend_index)
            current_friend = next_friend_index
            turn_number += 1
        losers = []
        for friend in friends:
            if friend not in visited_friends:
                losers.append(friend)
        return losers