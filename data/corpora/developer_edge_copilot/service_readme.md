# Service README

Service: bridge-sync service (bridge-sync)
Known issue: MQTT messages stop forwarding after token refresh in release 2.7.1

## First Checks

- compare the runtime value of mqtt_session_ttl with the documented baseline
- verify whether the token refresh handler rotates the cert cache before bridge-sync restarts
- restart bridge-sync only after exporting the local spool backlog metadata

## Never Do

Do not delete the local spool queue before exporting backlog metadata.
