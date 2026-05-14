# Banking Operations FAQs

## Service Level Agreements (SLAs)
Q: What is the SLA for tickets with a 'Critical' priority?
A: 'Critical' priority tickets require a 2-hour resolution window with an immediate manager alert.

Q: What is the SLA for 'High' priority tickets?
A: 'High' priority tickets have an 8-hour SLA.

Q: What is the SLA for 'Medium' priority tickets?
A: 'Medium' priority tickets have a 24-hour SLA.

Q: What is the SLA for 'Low' priority tickets?
A: 'Low' priority tickets have a 48-hour SLA.

Q: How quickly must a ticket with Regulatory Risk be resolved?
A: Any ticket with Regulatory Risk = 'Yes' must be resolved within 4 hours. Escalation must happen immediately to the primary compliance officers via the red-phone system.

## General Operations
Q: What steps should be taken if a customer reports unauthorized access to their mobile banking account?
A: Immediately lock the mobile banking profile, enforce a forced password reset, and dispatch a secure link to the customer's verified email. Flag the account for Level 2 Fraud Review.

Q: How do we handle international payment processing failures?
A: Check the SWIFT gateway logs for rejection codes. If the error is an invalid BIC/SWIFT code, notify the customer to correct the details. For compliance holds, route the ticket to the Anti-Money Laundering (AML) team.

Q: What is the procedure for a failed Core Banking System end-of-day batch job?
A: The IT Operations bridge must be alerted within 15 minutes of the failure. The batch job should be paused, the corrupted ledger entries rolled back, and the job re-initiated from the last successful checkpoint.

## Customer Data & KYC
Q: How long does it take to process a standard KYC (Know Your Customer) update?
A: Standard KYC updates submitted via the portal take 24-48 business hours to process.

Q: What happens if a customer's biometric authentication fails multiple times on the mobile app?
A: After 5 failed biometric attempts, biometric login is disabled, and the customer must authenticate using their PIN and a One-Time Password (OTP) sent to their registered device.

## ATM & Hardware
Q: What is the hardware replacement SLA for ATMs in metropolitan areas?
A: Critical hardware components for ATMs in metropolitan zones must be replaced within 12 hours of the reported failure.

Q: How do we troubleshoot ATM cash dispenser jams?
A: Attempt to run a remote diagnostic and mechanical reset cycle. If the jam remains un-cleared, the ATM must be marked 'Out of Service' and a regional field technician dispatched with an armored guard escort.

## Fraud & Security
Q: What should an agent do if they suspect a phishing attack on a customer?
A: Advise the customer to immediately change their login credentials. Freeze outward transactions temporarily, and report the phishing URL to the Cyber Threat Intelligence team to initiate domain takedown procedures.

Q: What defines an 'unusual high-volume transaction' for fraud detection?
A: Any transaction that exceeds 500% of the customer's average daily transfer limit or involves multiple rapid transfers to newly added overseas beneficiaries within a 2-hour window.

## Loans & Credit Cards
Q: A customer's loan disbursement is delayed. What should be checked?
A: Check the Loan Management System (LMS) for any pending manual underwriting approvals or compliance holds. Ensure the customer's linked savings account is active and not frozen.

Q: How are late fee reversals handled for credit cards?
A: Level 1 support can reverse one late fee per calendar year per customer up to $50. Anything exceeding this frequency or amount requires Level 2 escalation and manager approval.
