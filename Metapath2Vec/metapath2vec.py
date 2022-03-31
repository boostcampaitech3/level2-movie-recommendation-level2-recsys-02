from datareader import DataReader, Metapath2vecDataset
from skipgram import *

class Metapath2VecTrainer:
    def __init__(self, path):
        min_count, care_type = 0, 0
        batch_size, iterations = 50, 1
        window_size, dim, initial_lr = 10, 128, 0.025
        num_workers = 1
        
        self.data = DataReader(path, min_count, care_type)
        dataset = Metapath2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers, collate_fn=dataset.collate)
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = dim
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self, args):
        optimizer = optim.SparseAdam(list(self.skip_gram_model.parameters()), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

        for iteration in range(self.iterations):
            print("\n\n\nIteration: " + str(iteration + 1))
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 50000 == 0:
                        print(" Loss: " + str(running_loss))
        
        self.skip_gram_model.save_embedding(self.data.id2word, args.save_path)