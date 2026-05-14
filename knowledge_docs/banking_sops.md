# Banking Operations Standard Operating Procedures (SOPs)

## Regulatory Reporting SOP
Any ticket with Regulatory Risk = 'Yes' must be escalated immediately to the primary compliance officers via the red-phone system.

## ATM Alerts SOP
If an ATM is showing 'out of service' but the vault has sufficient cash:
1. Attempt to reset the hardware network adapter remotely.
2. If the remote reset fails, dispatch a regional field technician immediately.

## Credit Card Operations SOP
For credit limit increase requests that are approved but not reflecting on the customer's account:
1. Identify and clean the corrupted data records in the Core Banking System.
2. Schedule a rerun of the batch job overnight to sync the corrected records.

## Fraud Detection System SOP
If an account is suspected of fraud or exhibits unusually high-volume transactions:
1. Freeze the account immediately to prevent further unauthorized activity.
2. Contact the customer directly via their verified phone number to confirm the transactions.

## Loan Management SOP
If overdue loan notices are failing to generate:
1. Check the Document Generation Service for timeouts.
2. Restart the PDF microservice to resolve the timeout issue.
