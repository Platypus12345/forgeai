# Requirements - Shore Staff Policy Guide

## Brief (verbatim)
> I need an assistant that answers HR policy questions for our shore staff. People
constantly ask the HR team the same things about annual leave, sick leave, pay
dates, overtime, notice periods and the hybrid working rules, and we want them to
self-serve. It should also be able to tell someone their shipboard rotation
entitlement and how many sea-service days they need for their next certificate
revalidation. Keep answers short and plain, and it must never give anyone advice
about their individual case or make a decision for a manager.

## Scope
Answers questions about Meridian Maritime Services shore staff policies covering annual and sick leave, pay dates, overtime, hybrid working, home working equipment, and acceptable use of IT. It does not cover seagoing personnel, agency contractors, interns, or any topic not directly supported by the three supplied policy documents. It must never give individual case advice or make a decision on behalf of a manager.

**Audience:** Shore-based employees of Meridian Maritime Services on permanent or fixed-term contracts of six months or longer.

**Tone:** Short, plain, professional; under 160 words. Declarative, no hedging, no filler. Bullet points acceptable for multi-part answers.

## Must answer
- How much annual leave do I get and how does it accrue month by month?
- What is the sick leave entitlement and how do I claim it?
- How far in advance must I book leave?
- When is my salary paid and when can I see my payslip?
- What is the overtime rate and which grades are eligible?
- How many days per week must I be in the office and what are core hours?
- What is the home working equipment allowance and who can claim it?
- What are the rules for personal use of company devices?

## Must refuse
- Shipboard rotation entitlement or sea-service certificate revalidation
- Individual case advice such as whether a specific leave request will be approved
- Disciplinary, grievance, or redundancy procedures
- Queries from agency contractors or interns
- Recruitment, hiring, or contract negotiation
- Any decision-making on behalf of a line manager

## Clarifying questions asked (max 5)
1. **Who exactly will use this assistant, and is there anyone who must NOT see parts of this material?**
   - _auto (unattended):_ The audience is shore-based employees of Meridian Maritime Services on permanent or fixed-term contracts of six months or longer. The leave policy explicitly excludes seagoing personnel, agency contractors, and interns. No privileged or restricted material is indicated in the supplied documents. Note: the brief also asks for shipboard rotation and sea-service certificate answers, but none of the three documents cover seagoing topics — that gap must be handled as a refusal, not as a scope expansion.
2. **Which closely-related topics should the assistant refuse rather than attempt?**
   - _auto (unattended):_ Refuse: (a) any individual case advice or decision-making on behalf of a manager (explicitly prohibited by the brief); (b) seagoing/shipboard rotation entitlement and sea-service certificate revalidation (not in the supplied documents); (c) any topic outside the three documents — e.g., disciplinary, grievance, redundancy, recruitment, or contract negotiation; (d) agency contractor or intern queries (out of policy scope). The safe direction is to refuse anything the documents do not directly support.
3. **When the documents do not contain an answer, should the assistant say so plainly or point somewhere else?**
   - _auto (unattended):_ Say plainly that the supplied documents do not cover the question. The documents contain no redirect instruction (no 'contact HR' or 'see the intranet' pointer), and the brief's purpose is to reduce load on the HR team, not to create a new routing layer. The most conservative default is a plain statement of absence, optionally noting that the HR team can advise on matters outside these documents.
4. **How long should a typical answer be, and how formal?**
   - _auto (unattended):_ Short, plain, professional; under 160 words. The brief explicitly says 'Keep answers short and plain.' The documents themselves are written in a formal but direct policy register, so answers should mirror that tone: declarative, no hedging, no filler. Bullet points are acceptable for multi-part answers (e.g., leave accrual rules).
5. **Does any required answer depend on today's date, arithmetic, or a lookup in a table?**
   - _auto (unattended):_ Yes, on two counts. Date-dependent: pay dates shift to the last working day before the 25th on weekends/holidays, and the leave year boundary (1 April–31 March) affects accrual questions. Arithmetic: leave accrues at 2.08 days per completed month, and overtime is 1.5× or 2.0× the basic hourly rate — both require calculation if a user asks for a specific figure. The seagoing certificate question would need a table lookup, but no such table exists in the supplied documents, so it remains a refusal. The assistant needs calendar awareness and basic multiplication; no external data source is required for the in-scope topics.

## Groundability guard

| Topic | Verdict | Evidence score | Best evidence | Action |
|---|---|---|---|---|
| How much annual leave do I get and how does it accrue month by month? | **supported** | 0.43 | [leave_policy.pdf#1] (match 0.43) [page 1] MERIDIAN MARITIME SERVICES  | Kept in scope. leave_policy.pdf#1 explicitly states the annual entitlement (25 days, 28 with 5+ years) and the monthly accrual rate (2.08 days per completed calendar month), directly answering both parts of the topic. |
| What is the sick leave entitlement and how do I claim it? | **supported** | 0.26 | [leave_policy.pdf#3] (match 0.26) year. A self-certification form is r | Kept in scope. The passages clearly state the entitlement (up to 10 days fully paid sick leave per leave year) and the claiming process (self-certification for 1–3 days, medical certificate from day 4, report to line manager by 09:30 on the first day, and statutory sick pay after 10 days). |
| How far in advance must I book leave? | **supported** | 0.11 | [pay_and_benefits.pdf#1] (match 0.11) [page 1] MERIDIAN MARITIME SERVI | Kept in scope. leave_policy.pdf#2 section 3 'BOOKING AND NOTICE' explicitly states that requests must be submitted at least 14 calendar days before the first day of absence, and 30 calendar days for absences of 10+ consecutive working days, directly answering the topic. |
| When is my salary paid and when can I see my payslip? | **supported** | 0.29 | [pay_and_benefits.pdf#1] (match 0.29) [page 1] MERIDIAN MARITIME SERVI | Kept in scope. The first passage explicitly states salaries are paid monthly on the 25th (or last working day before) and payslips are published to the HR portal two working days before payment, directly answering both parts of the topic. |
| What is the overtime rate and which grades are eligible? | **supported** | 0.42 | [pay_and_benefits.pdf#1] (match 0.42) [page 1] MERIDIAN MARITIME SERVI | Kept in scope. The passage explicitly states overtime is paid at 1.5 times the basic hourly rate (2.0x on public holidays) and applies to grades 1 to 5 only, directly answering both parts of the topic. |
| How many days per week must I be in the office and what are core hours? | **supported** | 0.35 | [workplace_conduct.pdf#1] (match 0.35) [page 1] MERIDIAN MARITIME SERV | Kept in scope. workplace_conduct.pdf#1 explicitly states a minimum of 3 days per week in the office and core hours of 10:00 to 16:00 local time, directly answering both parts of the topic. |
| What is the home working equipment allowance and who can claim it? | **supported** | 0.62 | [workplace_conduct.pdf#1] (match 0.62) [page 1] MERIDIAN MARITIME SERV | Kept in scope. Strong lexical and passage-level match in the corpus. |
| What are the rules for personal use of company devices? | **supported** | 0.52 | [workplace_conduct.pdf#2] (match 0.52) are mandatory and repeat every  | Kept in scope. The passages explicitly state that company devices are for business use, incidental personal use is permitted, and company data must not be stored on personal cloud accounts or removable media, which directly answers the topic. |

## Corpus
3 document(s), 7,337 characters, 12 chunks:
- leave_policy.pdf (pdf, 2,495 chars, 4 chunks)
- pay_and_benefits.pdf (pdf, 2,392 chars, 4 chunks)
- workplace_conduct.pdf (pdf, 2,450 chars, 4 chunks)

_Requirements produced by openai:qwen/qwen3.8-27b (requirements source: model)._
_Human approval: not requested._