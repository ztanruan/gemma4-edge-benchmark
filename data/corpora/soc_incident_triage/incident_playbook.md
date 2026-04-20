# Incident Playbook

Host: finance-laptop-227
Detection: an encoded PowerShell launch followed by suspicious token refresh activity

## Analyst First Checks

- confirm the parent process chain for the encoded PowerShell execution
- pull recent identity-provider token refresh events for the affected user
- preserve local logs before any containment action changes device state

## Preserve

Do not delete local forensic artifacts or clear security logs before review.
