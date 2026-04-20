# Internal KB Article

Feature Area: tenant domain migration flow

## Customer Symptom

users lose SSO after the tenant primary domain is changed

## Support First Checks

- re-run domain verification for the new tenant primary domain
- reissue the SCIM provisioning secret to the identity provider
- keep the previous domain alias active for the documented transition window

## Do Not Suggest

Do not ask the customer to factory-reset network appliances as a first-line fix.
