from typing import List

class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        online_status = [True] * numberOfUsers
        offline_until = [-1] * numberOfUsers  # Track when each user will be back online

        for event in events:
            event_type, timestamp, detail = event
            timestamp = int(timestamp)

            # Process offline events first
            if event_type == "OFFLINE":
                user_id = int(detail)
                online_status[user_id] = False
                offline_until[user_id] = timestamp + 60

            # Process message events
            elif event_type == "MESSAGE":
                # Update online status for users who should come back online
                for user_id in range(numberOfUsers):
                    if not online_status[user_id] and timestamp >= offline_until[user_id]:
                        online_status[user_id] = True

                if detail == "ALL":
                    # Mention all users
                    for user_id in range(numberOfUsers):
                        mentions[user_id] += 1
                elif detail == "HERE":
                    # Mention only online users
                    for user_id in range(numberOfUsers):
                        if online_status[user_id]:
                            mentions[user_id] += 1
                else:
                    # Mention specific users
                    mentioned_ids = detail.split()
                    for mention in mentioned_ids:
                        if mention.startswith("id"):
                            user_id = int(mention[2:])  # Extract the user id
                            mentions[user_id] += 1

        return mentions