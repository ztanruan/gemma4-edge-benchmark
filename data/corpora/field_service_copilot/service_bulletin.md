# Field Service Bulletin

Customer Site: Meridian Cold Storage, Dock 4
Asset: ThermoGrid R9 controller (TG-R9-D4)

## Symptom

intermittent Modbus CRC errors after a controller update to build 3.14.2

## Technician First Checks

- inspect the shielded RS-485 cable termination and grounding
- lock the controller baud rate to 19200 to match the dock sensor bus
- confirm the last known-good site profile was restored after the update

## Do Not

Do not replace the compressor assembly before validating the controller and bus health.
