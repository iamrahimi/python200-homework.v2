Q1: Cloud computing is usually pay as you go: rent computing resources when needed. Owning servers means paying upfront and maintaining them yourself.

Q2: Vertical scaling means upgrading one machine; horizontal scaling means adding machines. The viral app and split-up data pipeline call for horizontal scaling. The faster-GPU model job calls for vertical scaling.

Q3 classifications: Gmail — SaaS; Azure Virtual Machines — IaaS; AWS S3 — IaaS storage; GitHub Codespaces — PaaS; Snowflake — managed data platform, often classified as SaaS; Supabase — BaaS. In your definitions, focus on how much infrastructure the provider manages versus what you manage.

Q4: Managed data platforms handle much of the setup and operation for data work. You gain convenience and built-in features; you give up some control and may become more tied to that provider.

Q5: A common case is a steady, predictable workload where owning and operating your own servers could cost less than paying cloud fees over time. Check whether this matches the lesson’s specific example.



Cloud Landscape
1. Hyperscalers
- AWS: Offers a very broad range of cloud services and is used by organizations from startups to large enterprises.
- Microsoft Azure: Works well with Microsoft tools and hybrid environments, so many large businesses use it.
- Google Cloud: Strong in data analytics, AI, and Kubernetes, which can suit data-focused organizations.
2. Azure to Supabase

I couldn’t open the lesson, so I can’t confirm the three specific reasons it gives. Check those points in the lesson before submitting. The switch may relate to Supabase being easier to set up, bundling services such as a database and authentication, or being a better fit for the course’s needs.

For my own project, I’d compare tools based on what the project needs, how hard they are to set up, and their cost—not choose by brand alone.

3. Service categories
1. Object storage: AWS S3 can store and retrieve image files.
2. GPU compute: An AWS EC2 GPU instance can run a training job, then shut down.
3. Serverless compute: AWS Lambda can run API code in response to requests and scale with demand.
4. LLM API: The OpenAI API accepts structured input and returns a text response.
4. Example stack

I’d build a bike commute dashboard, store uploaded CSV files in AWS S3, and use Supabase for the app’s database and user accounts. Using one provider can simplify billing and setup, but I might give up a service that better fits a particular part of the project and become more dependent on that provider.