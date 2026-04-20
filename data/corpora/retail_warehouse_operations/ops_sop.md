# Warehouse SOP

Operation: wave 7721 replenishment issue
Primary exception: pickers are hitting an empty forward slot while on-hand inventory still shows positive

## Required Checks

- cycle-count the reserve location before forcing an inventory correction
- review the latest short-pick adjustments for the affected SKU
- hold fragile orders in the active wave until slot accuracy is restored

## Prohibited Action

Do not force a negative inventory close to clear the exception quickly.
