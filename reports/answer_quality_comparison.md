# HR Policy & Employee Support Assistant
## Baseline vs Fine-Tuned Answer Quality Evaluation

**Evaluation date:** 2026-11-03
**Base model:** Qwen/Qwen2.5-0.5B-Instruct
**Fine-tuning method:** LoRA
**Test dataset:** data/splits/test.json
**Total held-out questions:** 15

## 1. Evaluation Objective

This evaluation compares the original Qwen model with the Qwen model adapted using LoRA. Both models were evaluated using the same held-out HR test questions.

## 2. Generation Settings

| Setting | Value |
|---|---|
| max_new_tokens | 120 |
| do_sample | False |

## 3. Answer-by-Answer Comparison

### Test Case 1: HR032

**Category:** attendance

**Employee question:**

How can I view my attendance history?

**Reference answer:**

Log in to the employee self-service portal and open the attendance or timesheet section. Select the relevant date range to review your records.

**Original Qwen answer:**

To view your attendance history, you typically need to log in to your HR system or access it through a web portal where you have access to your personal information. Here’s how you can do it:

1. **Log In**: Go to the HR system's login page on your computer or mobile device.

2. **Navigate to Attendance**: Once logged in, navigate to the section that displays your attendance details. This might be under "Attendance," "Leave," or similar sections depending on your HR system.

3. **View Attendance History**: Look for a section labeled as "Attendance" or something similar

**Fine-tuned Qwen + LoRA answer:**

Access your attendance record by logging into your account's designated portal or using the official attendance management system. Check for any updates or notifications related to your attendance status.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 34.977 | 3.113 |
| Peak GPU memory (MB) | 957.09 | 957.37 |

---

### Test Case 2: HR009

**Category:** leave_management

**Employee question:**

How can I apply for paternity leave?

**Reference answer:**

Contact HR or consult the employee handbook for the applicable paternity leave eligibility and procedure. Submit the request through the approved HR channel.

**Original Qwen answer:**

To apply for paternity leave, you should follow these steps:

1. **Review Your Employment Contract**: Ensure that your employment contract includes provisions for paternity leave. This typically outlines how much time you have to take off under certain circumstances.

2. **Contact HR**: Reach out to Human Resources (HR) at your organization. They will be able to provide detailed guidance on whether paternity leave is available and what documents you need to submit.

3. **Submit Documentation**: Based on the availability of paternity leave, you may need to submit additional documentation such as a medical certificate from a doctor confirming

**Fine-tuned Qwen + LoRA answer:**

Contact your manager or HR department to request paternity leave through their designated application process. Follow the official guidelines and submit all required documentation.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 31.546 | 1.805 |
| Peak GPU memory (MB) | 957.1 | 957.33 |

---

### Test Case 3: HR098

**Category:** company_policies

**Employee question:**

What is an employee privacy policy?

**Reference answer:**

An employee privacy policy explains how an organization collects, uses, stores, and protects employee personal information. Consult the official policy for specific rights and procedures.

**Original Qwen answer:**

An employee privacy policy outlines how your organization handles personal data of employees, including what types of data it collects, how it uses that data, and what measures are in place to protect it from unauthorized access, loss, or damage. Here’s a basic outline of what you might find in such a policy:

1. **Purpose**: Define the primary purpose for collecting and using employee data.

2. **Data Collection**: List all the data sources where your organization collects employee information (e.g., resumes, job applications, performance reviews).

3. **Data Use**: Describe how your organization will use this data

**Fine-tuned Qwen + LoRA answer:**

An employee privacy policy outlines how sensitive personal data is collected, stored, shared, and used by your organization. It should be clear about who has access, what they can do with their data, and how it will be protected. Contact HR if you need more details.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 28.352 | 3.691 |
| Peak GPU memory (MB) | 957.05 | 957.63 |

---

### Test Case 4: HR004

**Category:** leave_management

**Employee question:**

Can I cancel an approved leave request?

**Reference answer:**

Contact your manager or HR and follow the official leave cancellation process. Update the request in the HR system if that option is available.

**Original Qwen answer:**

Yes, you can typically cancel an approved leave request if it's within the specified time frame after approval. Typically, this means that your leave request has been approved by your supervisor or manager, and you have a set timeframe for submitting a new request. Once you submit the new request, it will be processed according to the original approval process.

If there's any delay in processing your new request due to unforeseen circumstances (like a work-related emergency), you should inform your supervisor or manager as soon as possible so they can take appropriate action. It’s important to keep all communications open and responsive throughout the

**Fine-tuned Qwen + LoRA answer:**

Yes, you can cancel a leave request by contacting your manager or HR. Follow the cancellation process outlined in the approved leave approval documentation.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 18.086 | 1.781 |
| Peak GPU memory (MB) | 957.09 | 957.33 |

---

### Test Case 5: HR040

**Category:** attendance

**Employee question:**

How do I request a shift change?

**Reference answer:**

Discuss the proposed change with your manager and submit a request through the organization's scheduling or HR process. A shift change requires the appropriate approval.

**Original Qwen answer:**

To request a shift change in your workplace, follow these steps:

1. **Prepare Your Shift**: Ensure you have all necessary documents ready such as your ID, work permit, and any relevant paperwork.

2. **Contact HR**: Reach out to Human Resources (HR) for assistance. You can usually find their contact details on the company's website or through HR support services.

3. **Request a Shift Change**: Once at HR, they will guide you through the process. Typically, this involves filling out a form with your name, department, and reason for requesting a shift change. They may also

**Fine-tuned Qwen + LoRA answer:**

Contact your manager or HR for details on requesting a shift change. Follow the approved process and follow-up with the designated person.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 6.495 | 1.667 |
| Peak GPU memory (MB) | 957.09 | 957.3 |

---

### Test Case 6: HR059

**Category:** payroll

**Employee question:**

What is a reimbursement?

**Reference answer:**

A reimbursement is repayment of an eligible expense incurred by an employee on behalf of the organization, subject to policy, approval, and supporting documentation.

**Original Qwen answer:**

A reimbursement is a financial transfer from one entity (the reimbuster) to another entity (the recipient) for expenses incurred by the recipient in accordance with the terms of the agreement between them. Reimbursements can be used for various purposes such as covering medical bills, paying for travel expenses, or providing compensation for work-related injuries.

Reimbursement typically involves two parties: the reimbuster and the recipient. The reimbuster provides the expense, while the recipient pays it back according to the agreed-upon terms. This process helps ensure that expenses are accurately recorded and accounted for, maintaining transparency and accountability

**Fine-tuned Qwen + LoRA answer:**

A reimbursement is a payment made by your organization for expenses incurred through approved authorized transactions. It includes payments from internal accounts, external vendors, or other approved sources. Contact HR if you need help with reimbursement requests.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 6.458 | 2.701 |
| Peak GPU memory (MB) | 957.0 | 957.47 |

---

### Test Case 7: HR060

**Category:** payroll

**Employee question:**

How do I check the status of a reimbursement?

**Reference answer:**

Open the expense or reimbursement section of your employee portal and review the request status. Contact the finance or payroll team if more information is needed.

**Original Qwen answer:**

To check the status of a reimbursement, you can typically follow these steps:

1. **Review Your Receipt**: First, ensure that your receipt for the reimbursement has been received by the department responsible for processing reimbursements.

2. **Check with HR**: Contact Human Resources (HR) directly. They will have access to all incoming and outgoing communications related to reimbursements and can provide detailed information about the status of each reimbursement request.

3. **Use the Reimbursement System**: Many organizations use internal systems like SAP, Oracle, or Microsoft Dynamics to manage reimbursements. Check if there's a dedicated section

**Fine-tuned Qwen + LoRA answer:**

Contact your manager or HR for the reimbursement status. Check the official reimbursement portal or use the official reporting system.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 5.692 | 1.386 |
| Peak GPU memory (MB) | 957.13 | 957.28 |

---

### Test Case 8: HR036

**Category:** attendance

**Employee question:**

How is work-from-home attendance recorded?

**Reference answer:**

Use the organization's approved remote attendance or timesheet system. Follow the instructions provided by your manager or HR.

**Original Qwen answer:**

Work-from-home attendance can be recorded in several ways depending on your organization's policies and practices:

1. **Regular Attendance Records**: Many companies keep detailed records of employees' regular hours worked. This might include entries for each day they were at home.

2. **Attendance Apps/Software**: Some companies use apps or software that track attendance. These tools often allow you to set up automatic reminders for when you need to be present.

3. **Employee Self-Report Forms**: Some organizations require employees to fill out self-report forms detailing their daily activities. This could include times spent working from home.

4

**Fine-tuned Qwen + LoRA answer:**

Work-from-home employees should record their attendance using the approved system or method provided by their organization. Check with your manager or HR for any required documentation.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 5.908 | 1.915 |
| Peak GPU memory (MB) | 957.09 | 957.35 |

---

### Test Case 9: HR078

**Category:** company_policies

**Employee question:**

Where can I find company policies?

**Reference answer:**

Check the official employee handbook, internal knowledge portal, or HR system. Contact HR if you cannot locate the latest approved policy document.

**Original Qwen answer:**

Company policies are typically found in several places:

1. **Company Manual**: This is usually located on the first floor of your office building or in a central location within the company.
2. **Employee Handbook**: This document outlines all company policies and procedures for employees.
3. **HR Department Website**: Many companies have their own website where they provide detailed information about their policies and procedures.
4. **Internal Communication Channels**: Regularly scheduled meetings with HR or other relevant departments can also be useful for getting updates on company policies.

If you need more specific guidance on where to find certain policies, it might

**Fine-tuned Qwen + LoRA answer:**

Company policies are typically found in the organization's official website, HR portal, or designated internal communication channels. Contact your manager or HR if you need help locating them.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 6.217 | 2.06 |
| Peak GPU memory (MB) | 957.05 | 957.37 |

---

### Test Case 10: HR014

**Category:** leave_management

**Employee question:**

What should I do if I become sick during a holiday?

**Reference answer:**

Check the organization's sick leave policy to understand how illness during a holiday is handled. Notify your manager or HR and submit supporting information if required.

**Original Qwen answer:**

If you become ill while on vacation, it's important to take care of your health as much as possible. Here are some steps you can consider:

1. **Stay Hydrated**: Drink plenty of water to stay hydrated. Avoid sugary drinks that might dehydrate you.

2. **Rest**: Ensure you get adequate rest. Rest is crucial for recovery.

3. **Eat Well**: Eat nutritious meals to help your body recover faster. Avoid foods that may make you feel worse.

4. **Hydrogen Peroxide**: Use hydrogen peroxide to clean yourself and your surroundings. It

**Fine-tuned Qwen + LoRA answer:**

Contact your employer's designated sick leave coordinator or HR for guidance on how to handle illness during holidays.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 5.62 | 1.315 |
| Peak GPU memory (MB) | 957.18 | 957.28 |

---

### Test Case 11: HR057

**Category:** payroll

**Employee question:**

What should I do if my salary is not credited?

**Reference answer:**

Check the scheduled payroll date and your bank account details. If the salary remains uncredited, contact payroll or HR through the official support channel.

**Original Qwen answer:**

If your salary is not credited, it's important to understand that this could be due to several reasons such as a discrepancy between your account balance and the actual amount of money you earned. Here’s how you can handle this situation:

1. **Check Your Account Balance**: Ensure that your bank statement shows the correct amount of your salary. Sometimes, there might be a small error in the transaction.

2. **Contact Your Bank**: Reach out directly to your bank to inquire about any discrepancies. They may need to verify your account details or provide additional documentation.

3. **Review Your Payroll Statement**:

**Fine-tuned Qwen + LoRA answer:**

Check your pay stubs for any errors or discrepancies. Contact payroll support or HR if you need further assistance.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 5.764 | 1.365 |
| Peak GPU memory (MB) | 957.14 | 957.3 |

---

### Test Case 12: HR079

**Category:** company_policies

**Employee question:**

What is a code of conduct?

**Reference answer:**

A code of conduct describes expected professional behaviour, ethical standards, workplace responsibilities, and rules employees are expected to follow.

**Original Qwen answer:**

A Code of Conduct is a set of rules that organizations establish for their employees to ensure they act in accordance with ethical standards and values. These codes typically outline expectations regarding behavior, professionalism, integrity, and respect for colleagues and customers.

The purpose of a Code of Conduct is to provide a framework for maintaining a positive work environment where employees can operate without fear of retaliation or discrimination. It helps prevent unethical practices, promotes fairness, and fosters trust among team members.

Organizations often use these codes as guidelines for performance evaluations, promotions, and disciplinary actions. They also serve as a reminder to all employees about

**Fine-tuned Qwen + LoRA answer:**

A code of conduct outlines expectations for behavior, performance, and adherence to organizational values. It can be shared with employees through formal channels such as email, intranet, or meeting minutes.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 5.854 | 2.383 |
| Peak GPU memory (MB) | 957.05 | 957.44 |

---

### Test Case 13: HR073

**Category:** onboarding

**Employee question:**

How can I get help setting up my work laptop?

**Reference answer:**

Contact the IT helpdesk or onboarding coordinator for device setup instructions. Follow the organization's security and software installation guidelines.

**Original Qwen answer:**

Setting up a work laptop can be quite straightforward if you follow these steps:

1. **Check Your Laptop**: Before starting any setup process, make sure your work laptop is properly charged and that all necessary cables are connected.

2. **Connect Power Supply**: Plug in the power supply unit (PSU) of your laptop into the wall outlet. Ensure it’s plugged in correctly before turning on the laptop.

3. **Turn On the Laptop**: Press and hold the power button for about 5 seconds until the laptop lights up with a green light. This indicates that the battery is fully charged.

4

**Fine-tuned Qwen + LoRA answer:**

Contact your IT department for assistance. Check the official support documentation or online resources provided by the organization.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 6.074 | 1.283 |
| Peak GPU memory (MB) | 957.14 | 957.28 |

---

### Test Case 14: HR018

**Category:** leave_management

**Employee question:**

What documents are needed for sick leave?

**Reference answer:**

Required documents depend on the organization's sick leave policy and the duration or circumstances of the absence. Ask HR which documents are applicable.

**Original Qwen answer:**

For sick leave, you typically need a valid medical certificate from your healthcare provider confirming that you have been diagnosed with a work-related illness or injury. This document should be provided by your employer's human resources department or directly by your doctor. Additionally, it may also be necessary to provide proof of your attendance at work during the period when you are unable to perform your regular duties due to illness.

It's important to note that some employers might require additional documentation such as a letter from your employer stating that you are unable to work because of illness or injury. Always ensure that all required documentation is in order before

**Fine-tuned Qwen + LoRA answer:**

Check your organization's sick-leave policy and follow the required documentation requirements. Contact HR if you need help with filling out the necessary forms.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 6.813 | 2.037 |
| Peak GPU memory (MB) | 957.09 | 957.33 |

---

### Test Case 15: HR065

**Category:** onboarding

**Employee question:**

What happens during employee onboarding?

**Reference answer:**

Onboarding introduces a new employee to the organization, its policies, workplace tools, team, responsibilities, and required administrative procedures.

**Original Qwen answer:**

During employee onboarding, a company typically provides several key steps to help new hires quickly integrate into their roles and understand the organization's culture, processes, and expectations. Here’s what you might expect:

1. **Introduction to Company Culture**: The first step is often to introduce the company culture through team-building activities, workshops, or presentations that highlight the values and goals of the organization.

2. **Onboarding Process**: This includes training sessions where new employees learn about the job responsibilities, procedures, and how to interact with colleagues and management. It also involves orientation meetings where they can familiarize themselves with

**Fine-tuned Qwen + LoRA answer:**

Onboarding typically includes training sessions, orientation, and a review of organizational policies and procedures. Contact your manager for more details.

| Performance metric | Baseline | Fine-tuned |
|---|---:|---:|
| Latency (seconds) | 5.885 | 1.792 |
| Peak GPU memory (MB) | 957.05 | 957.28 |

---

## 4. Overall Performance Summary

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Average latency (seconds) | 11.983 | 2.02 |
| Average peak GPU memory (MB) | 957.09 | 957.36 |
| Total test questions | 15 | 15 |

## 5. Qualitative Review

The generated answers should be manually reviewed for:

- Relevance to the employee question.
- Correctness compared with the reference answer.
- Completeness of the response.
- Unsupported policy claims or hallucinations.
- Clear and professional HR communication.

**Note:** Lower training loss or response latency alone does not prove improved answer quality. No automatic quality score is assigned by this report.

## 6. Evaluation Limitations

- The evaluation uses 15 held-out synthetic HR examples.
- The dataset does not represent an actual company's official HR policy handbook.
- Manual review is required before claiming a quality improvement.
- Results may vary with hardware load and generation conditions.

## 7. Conclusion

This report records the baseline and LoRA fine-tuned model outputs and their measured performance. Final conclusions about answer quality must be based on review of the individual test cases.